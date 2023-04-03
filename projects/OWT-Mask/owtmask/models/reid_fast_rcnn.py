# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import select_foreground_proposals

__all__ = ["fast_rcnn_inference", "ReidFastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    objectness_scores: List[torch.Tensor],
    embeds: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, objectness_per_image, embeds_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        ) for scores_per_image, objectness_per_image, boxes_per_image, embeds_per_image, image_shape in zip(scores, objectness_scores, boxes, embeds, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]  # list[results], list[filter_inds]


def fast_rcnn_inference_single_image(
    boxes,
    scores,  # [N, C+1]
    objectness_scores,
    embeds,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """

    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    bg_scores = scores[:, -1]  # OWT
    scores = scores[:, :-1]
    num_classes = scores.shape[1]  # OWT
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]  # [ATTN] 这里可能复制多个框 但是我这里就一个类别就无所谓了
    else:
        boxes = boxes[filter_mask]
    # print('debug: scores_pre.shape = {}, filter_mask.shape = {}'.format(scores.shape, filter_mask.shape))
    scores = scores[filter_mask]  # [ATTN] 这里score 必须是 1d 因为后面的 nms要求是1d
    # print("scores.shape = {}".format(scores.shape))
    # reid_dim = embeds.shape[-1]
    # embeds = embeds.view(-1, 1, reid_dim).repeat(1, 80, 1)  # [ATTN] 80类时候推理 img_demo的时候需要 但是track inference的时候不需要
    # embeds = embeds.view(-1, 1, reid_dim)
    # embeds = embeds[filter_mask]
    # print('embeds.shape = {}'.format(embeds.shape))  # [1000, 256]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)  # [TODO] 为了和IDOL统一，这里只给出top300的instances 后续进行过滤等处理
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    if num_classes == 1:
        boxes, scores, embeds, filter_inds = boxes[keep], scores[keep], embeds[keep], filter_inds[keep]
        # print('debug: infer, bg_scores.shape = {}, keep.shape = {}, keep = {}'.format(bg_scores.shape, keep.shape, keep))
        bg_scores = bg_scores[keep]
        objectness_scores = objectness_scores[keep]
    elif num_classes == 80:
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        # OWT: Find out which row are kept in the original scores tensor (1000, 80+1),
        # and select the corresponding bg_scores
        keep_row = torch.div(keep, num_classes, rounding_mode='trunc')
        bg_scores = bg_scores[keep_row]
        objectness_scores = objectness_scores[keep_row]
        embeds = embeds[keep_row]


    # print('debug: inference, topk_per_image = {}, boxes.shape = {}'.format(topk_per_image, boxes.shape))
    # debug: inference, topk_per_image = 100, boxes.shape = torch.Size([7, 4])  原始只保留了很少的部分
    # print('debug: scores.shape = {}'.format(scores.shape))  # 这是的scores是1d的 在外面进行升维需要
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)  # [ATTN]
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.embeds = embeds
    result.bg_scores = bg_scores
    result.objectness = objectness_scores
    return result, filter_inds[:, 0]


class ReidFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        use_fed_loss: bool = False,
        use_sigmoid_ce: bool = False,
        get_fed_loss_cls_weights: Optional[Callable] = None,
        fed_loss_num_classes: int = 50,
        reid_feat_dim=256,
        use_deformable_reid_head=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)

        self.reid_feat_dim = reid_feat_dim
        self.use_deformable_reid_head = use_deformable_reid_head
        
        if self.use_deformable_reid_head:
            pass
            # nheads = cfg.MODEL.DDETRS.NHEADS
            # dropout = cfg.MODEL.DDETRS.DROPOUT
            # dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
            # dec_n_points = cfg.MODEL.DDETRS.DEC_N_POINTS
            # num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
            # decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
            # dropout, "relu", num_feature_levels, nheads, dec_n_points)
            # self.reid_embed_head = nn.ModuleList([
            #     DeformableReidHead(hidden_dim, decoder_layer, cfg.N_LAYER_DEFORMABLE_REID),  # [ATTN]
            #     MLP(hidden_dim, hidden_dim, hidden_dim, 3)]
            # )
        else:
            self.reid_embed_head = MLP(input_size, reid_feat_dim, reid_feat_dim, 3)  # [TODO] 是否需要初始化

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.use_fed_loss = use_fed_loss
        self.use_sigmoid_ce = use_sigmoid_ce
        self.fed_loss_num_classes = fed_loss_num_classes
        

        if self.use_fed_loss:
            assert self.use_sigmoid_ce, "Please use sigmoid cross entropy loss with federated loss"
            fed_loss_cls_weights = get_fed_loss_cls_weights()
            assert (
                len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"            : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"         : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"               : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},  # noqa
            "use_fed_loss"              : cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            "use_sigmoid_ce"            : cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            "get_fed_loss_cls_weights"  : lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER),  # noqa
            "fed_loss_num_classes"      : cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES,
            "reid_feat_dim"             : cfg.MODEL.ROI_HEADS.REID_FEAT_DIM,
            "use_deformable_reid_head"  : cfg.MODEL.ROI_HEADS.USE_DEFORMABLE_REID_HEAD,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)  # 这里是多张图片展平了一起inference的
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        reid_embeds = self.reid_embed_head(x)
        return scores, proposal_deltas, reid_embeds

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, reid_embeds = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        losses.update(self.loss_asso(predictions, proposals))
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def loss_asso(self, predictions, proposals):
        losses = {}
        num_props_per_img = [len(_) for _ in proposals] 
        fg_proposals, fg_selection_masks = select_foreground_proposals(  # proposals已经筛过了 不一定是512了
                    proposals, self.num_classes
                )  

        # compute reid loss for foregrond proposals
        _, _, reid_embeds = predictions
        reid_embeds = reid_embeds.split(num_props_per_img, dim=0)
        posi_embeds = [reid_embed[fg_selection_mask] for reid_embed, fg_selection_mask in zip(reid_embeds, fg_selection_masks)]
        back_embeds = [reid_embed[~fg_selection_mask] for reid_embed, fg_selection_mask in zip(reid_embeds, fg_selection_masks)]
        loss_co = 0
        loss_aux = 0
        B = len(fg_proposals)
        pairs = 0
        for bid in range(0,B,2):
            key_reid_embed = posi_embeds[bid]
            ref_reid_embed = posi_embeds[bid + 1]
            # back_reid_embed = torch.cat([back_embeds[bid], back_embeds[bid + 1]])
            back_reid_embed = back_embeds[bid + 1]
            if key_reid_embed.shape[0] == 0 or ref_reid_embed.shape[0] == 0:
                loss_co += key_reid_embed.sum() * 0
                loss_aux += key_reid_embed.sum() * 0
                continue
            pairs += 1

            key_gt_ids = fg_proposals[bid].gt_ids
            ref_gt_ids = fg_proposals[bid+1].gt_ids 
            assert (key_gt_ids == -1).sum() == 0 and (ref_gt_ids == -1).sum() == 0
            contrast = torch.einsum('nc,kc->nk',[key_reid_embed,torch.cat([ref_reid_embed, back_reid_embed])])
            mask = key_gt_ids.view(-1, 1) == ref_gt_ids.view(1, -1)
            mask = torch.cat([mask,torch.zeros(key_reid_embed.shape[0], back_reid_embed.shape[0]).to(mask)], dim=1)  

            contrast_exp = torch.exp(contrast)
            log_prob = contrast - torch.log(contrast_exp.sum(dim=1, keepdim=True) + 1e-12)
            mean_log_prob = (log_prob * mask).sum(1) / (mask.sum(1) + 1e-12)
            loss_co += mean_log_prob.mean() * (-1.0)

            # aux loss
            aux_key = nn.functional.normalize(key_reid_embed, dim=1)
            aux_ref = nn.functional.normalize(torch.cat([ref_reid_embed, back_reid_embed]), dim=1)
            cosine = torch.einsum('nc,kc->nk',[aux_key,aux_ref])
            loss_aux += (torch.abs(cosine - mask.float())**2).mean()
        if pairs == 0:
            print('debug: train, pairs = {}'.format(pairs))
            losses['loss_co'] = posi_embeds[0].sum() * 0
            losses['loss_aux'] = posi_embeds[0].sum() * 0
        else:
            losses['loss_co'] = loss_co / pairs  # [ATTN] 可能为0
            losses['loss_aux'] = loss_aux / pairs
        return losses

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/fed_loss.py  # noqa
    # with slight modifications
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes

        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
    # with slight modifications
    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        N = pred_class_logits.shape[0]
        K = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(N, K + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :K]

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction="none"
        )

        if self.use_fed_loss:
            fed_loss_classes = self.get_fed_loss_classes(
                gt_classes,
                num_fed_loss_classes=self.fed_loss_num_classes,
                num_classes=K,
                weight=self.fed_loss_cls_weights,
            )
            fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
            fed_loss_classes_mask[fed_loss_classes] = 1
            fed_loss_classes_mask = fed_loss_classes_mask[:K]
            weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()
        else:
            weight = 1

        loss = torch.sum(cls_loss * weight) / N
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]  # [ATTN]  会筛选正样本去求box loss
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        # add object+bg as recall cue
        softmax = torch.nn.Softmax(dim=0)
        objectness_scores = [softmax(proposals[0].objectness_logits)]

        embeds = self.predict_embeds(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        # print('debug: image_shapes = {}'.format(image_shapes))
        return fast_rcnn_inference(
            boxes,
            scores,
            objectness_scores,
            embeds,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)
    
    def predict_embeds(
        self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        _, _, embeds = predictions
        num_prop_per_image = [len(p) for p in proposals]
        return embeds.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:  # false
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        # print('proposals.shapelist = {}'.format([_.objectness_logits.shape for _ in proposals]))
        # [TODO] 在这里直接把 scores 替换成sqrt(bg* obj)
        # objs = torch.cat([_.objectness_logits for _ in proposals])
        # print('probs.shape = {}, obj.shape = {}, max = {}, min = {}'.format(probs.shape, objs.shape, objs.max(), objs.min()))
        # # print('ce = {}'.format(self.use_sigmoid_ce))
        # # objs = objs.sigmoid()
        # objs = torch.nn.Softmax(dim=0)(objs)
        # # probs[:,0] = torch.sqrt(probs[:,-1] * objs * objs.shape[0])
        # probs[:,0] = torch.sqrt(probs[:,-1] * objs * 1000)
        # # print('probs.shape = {}'.format(probs.shape))

        # [TODO] 注释下面两行即可
        # objs = torch.cat([torch.nn.Softmax(dim=0)(proposals[i].objectness_logits) * proposals[i].objectness_logits.shape[0] for i in range(len(proposals))])
        # probs[:, 0] = torch.sqrt(probs[:,-1] * objs)
        return probs.split(num_inst_per_image)
