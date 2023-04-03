# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.structures import Boxes, ImageList, Instances


from detectron2.modeling import (
    ROI_HEADS_REGISTRY, ROIHeads, build_mask_head, ResNet,
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.backbone import BottleneckBlock
from detectron2.modeling.roi_heads import select_foreground_proposals


from .reid_fast_rcnn import ReidFastRCNNOutputLayers, fast_rcnn_inference

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class ReidRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    See :paper:`ResNet` Appendix A.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: nn.Module,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        coco_pretrain = True,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        if isinstance(res5, (list, tuple)):
            res5 = nn.Sequential(*res5)
        self.res5 = res5
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None
        if self.mask_on:
            self.mask_head = mask_head

        self.coco_pretrain = coco_pretrain
    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        mask_on           = cfg.MODEL.MASK_ON
        coco_pretrain     = cfg.INPUT.COCO_PRETRAIN
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret['coco_pretrain'] = coco_pretrain
        # Compatbility with old moco code. Might be useful.
        # See notes in StandardROIHeads.from_config
        if not inspect.ismethod(cls._build_res5_block):
            logger.warning(
                "The behavior of _build_res5_block may change. "
                "Please do not depend on private methods."
            )
            cls._build_res5_block = classmethod(cls._build_res5_block)

        ret["res5"], out_channels = cls._build_res5_block(cfg)
        ret["box_predictor"] = ReidFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)  # [TODO]
        )

        if mask_on:
            ret["mask_head"] = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[Boxes]):
        # print('features.shape = {}, boxes.shape = {}'.format(features[0].shape, boxes[0].tensor.shape))
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets) # [ATTN] 往proposals里面补充一个gt_ids
        del targets
        proposal_boxes = [x.proposal_boxes for x in proposals]
        num_props_per_img = [len(_) for _ in proposals] 
        # print('debug: train, proposal_boxes.shape list = {}'.format([_.tensor.shape for _ in proposal_boxes]))  # 这个类型是Box 需要.tensor才能取到值
        # debug: train, proposal_boxes.shape list = [torch.Size([512, 4]), torch.Size([512, 4]), torch.Size([512, 4]), torch.Size([512, 4])]
        # [TODO] 下面这个会超内存
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # box_features_list = []
        # for i in range(len(proposal_boxes)):
        #     box_features_list.append(self._shared_roi_transform(
        #         [features[f][i:i+1] for f in self.in_features], proposal_boxes[i: i+1]
        #     ))
        
        # box_features = torch.cat(box_features_list)
        
        # print('debug: train, self.in_features = {}'.format(self.in_features))
        # debug: train, self.in_features = ['res4']
        # print('debug: train, box_features.shape = {}'.format(box_features.shape))
        # debug: train, box_features.shape = torch.Size([2048, 2048, 7, 7])
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))  # [torch.Tensor, torch.Tensor, torch.Tensor]

        # print('debug: inference, predictions.shapelist = {}'.format([_.shape for _ in predictions]))
        # debug: inference, predictions.shapelist = [torch.Size([3000, 2]), torch.Size([3000, 4]), torch.Size([3000, 256])]  # 300 * 10

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)  # [ATTN] prediction已经将batch整合了 proposals还没有
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(  # proposals已经筛过了 不一定是512了
                    proposals, self.num_classes
                )  
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]  # box_features.shape = [batch_total_proposals, C, 7, 7]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

                # compute reid loss for foregrond proposals
                _, _, reid_embeds = predictions
                # print('debug: train, reid_embeds.shape = {}'.format(reid_embeds.shape))
                # debug: train, reid_embeds.shape = torch.Size([2048, 256])
                # print('debug: train, fg_selection_masks.shape list = {}'.format([_.shape for _ in fg_selection_masks]))
                # debug: train, fg_selection_masks.shape list = [torch.Size([512]), torch.Size([512]), torch.Size([512]), torch.Size([512])]
                reid_embeds = reid_embeds.split(num_props_per_img, dim=0)
                posi_embeds = [reid_embed[fg_selection_mask] for reid_embed, fg_selection_mask in zip(reid_embeds, fg_selection_masks)]
                back_embeds = [reid_embed[~fg_selection_mask] for reid_embed, fg_selection_mask in zip(reid_embeds, fg_selection_masks)]
                loss_co = 0
                loss_aux = 0
                B = len(proposals)
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
                    # print('debug: train, key_reid_embed.shape = {}, ref_reid_embed.shape = {}'.format(key_reid_embed.shape, ref_reid_embed.shape))
                    # debug: train, key_reid_embed.shape = torch.Size([2, 256]), ref_reid_embed.shape = torch.Size([0, 256])
                    # print('debug: train, proposals[bid]._fields.keys() = {}'.format(proposals[bid+1]._fields.keys()))
                    # debug: train, proposals[bid]._fields.keys() = dict_keys(['proposal_boxes', 'objectness_logits', 'gt_classes', 'gt_boxes', 'gt_masks', 'gt_ids'])  
                    # 为什么一开始会报错说没有gt_ids 因为如果一张图中没有正样本 则不会塞其他的gt信息 只有gt_labels且全为负样本
                    key_gt_ids = proposals[bid].gt_ids
                    ref_gt_ids = proposals[bid+1].gt_ids 
                    assert (key_gt_ids == -1).sum() == 0 and (ref_gt_ids == -1).sum() == 0
                    contrast = torch.einsum('nc,kc->nk',[key_reid_embed,torch.cat([ref_reid_embed, back_reid_embed])])
                    mask = key_gt_ids.view(-1, 1) == ref_gt_ids.view(1, -1)
                    mask = torch.cat([mask,torch.zeros(key_reid_embed.shape[0], back_reid_embed.shape[0]).to(mask)], dim=1)  # 为什么这里面mask.float()会nan[?]
                    # print('debug: train, mask.sum() = {}'.format(mask.sum()))
                    # print('debug: train, contrast.shape = {}, mask.shape = {}'.format(contrast.shape, mask.shape))
                    # debug: train, contrast.shape = torch.Size([55, 40]), mask.shape = torch.Size([55, 40])
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
            return [], losses
        else:
            if self.coco_pretrain:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)  # list[result], list[filter_inds]
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
                return pred_instances, {}
            else:
                scores =  self.box_predictor.predict_probs(predictions, proposals)  #[TODO] 这地方原来多了一个','结果导致scores多包装了一层 []
                boxes = self.box_predictor.predict_boxes(predictions, proposals)  # list[tensor]
                embeds = self.box_predictor.predict_embeds(predictions, proposals)
                softmax = torch.nn.Softmax(dim=0)
                objectness_scores = [softmax(props.objectness_logits) for props in proposals]  # 这里可能是一个clip_len 而不是一张图
                # print('debug: video inference, scores[0].shape = {}'.format(scores[0].shape))
                image_shapes = [x.image_size for x in proposals]
                # print('image_shapes = {}'.format(image_shapes))  # image_shapes = [(480, 640)] 所以proposals确实长度为1
                # 使用track额外的推理参数
                pred_instances, _ = fast_rcnn_inference(boxes, scores, objectness_scores, embeds, image_shapes, 0.0, 1.0, 1000)  # 主要是为了获得Instance类 过后面的Mask_head  里面不做任何过滤，因为rpn输入限制为了300
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
                return pred_instances
                # output = {
                #     'pred_logits': torch.stack([_.scores.view(-1, 1) for _ in pred_instances]), # [B, N, C]  [ATTN] 给scores升维
                #     'pred_boxes': torch.stack([_.pred_boxes.tensor for _ in pred_instances]),  # 这个是Boxes类
                #     'pred_inst_embed': torch.stack([_.embeds for _ in pred_instances]),
                #     'pred_masks': torch.stack([_.pred_masks for _ in pred_instances]),
                # }
                # return output

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            feature_list = [features[f] for f in self.in_features]
            # print("len_features = {}, len_instances = {}".format(len(feature_list), len(instances)))
            x = self._shared_roi_transform(feature_list, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances
