# Copyright (c) Facebook, Inc. and its affiliates.
import os
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling import Backbone, build_backbone, detector_postprocess, build_proposal_generator, build_roi_heads, META_ARCH_REGISTRY
import torchvision.ops as ops


from .models.tracker import IDOL_Tracker
from owt_scipts.owt_utils import store_TAOnpz


__all__ = ["OWTMASK"]


@META_ARCH_REGISTRY.register()
class OWTMASK(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        batch_infer_len: int = 10,
        coco_pretrain: True,
        inference_fw: True,
        inference_tw: True,
        memory_len: int = 3,
        inference_select_thres: 0.1,
        temporal_score_type: str,
        is_multi_cls: True,
        apply_cls_thres: 0.05,
        output_dir: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

        self.batch_infer_len = batch_infer_len
        self.coco_pretrain = coco_pretrain
        self.inference_tw = inference_tw
        self.inference_fw = inference_fw
        self.memory_len = memory_len
        self.inference_select_thres = inference_select_thres
        self.temporal_score_type = temporal_score_type
        self.apply_cls_thres = apply_cls_thres
        self.is_multi_cls = is_multi_cls
        self.output_dir = output_dir

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "batch_infer_len": cfg.MODEL.OWTMASK.BATCH_INFER_LEN,
            "coco_pretrain": cfg.INPUT.COCO_PRETRAIN,
            "inference_fw": cfg.MODEL.OWTMASK.INFERENCE_FW,
            "inference_tw": cfg.MODEL.OWTMASK.INFERENCE_TW,
            "memory_len": cfg.MODEL.OWTMASK.MEMORY_LEN,
            "inference_select_thres": cfg.MODEL.OWTMASK.INFERENCE_SELECT_THRES,
            "temporal_score_type": cfg.MODEL.OWTMASK.TEMPORAL_SCORE_TYPE,
            "is_multi_cls": cfg.MODEL.OWTMASK.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.OWTMASK.APPLY_CLS_THRES,
            "output_dir": cfg.OUTPUT_DIR,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.training:
            images = self.preprocess_image(batched_inputs)
            gt_instances = []
            for video in batched_inputs:
                for frame in video["instances"]:
                    gt_instances.append(frame.to(self.device))
            gt_instances = self.prepare_targets(gt_instances)  # 过滤empty instance  filter gt_ids=-1

            features = self.backbone(images.tensor)

            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)  # [ATTN]
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        elif self.coco_pretrain:
            return self.coco_inference(batched_inputs)
        else:
            images = self.preprocess_image(batched_inputs)
            video_len = len(batched_inputs[0]['file_names'])
            clip_length = self.batch_infer_len  # 10                
            pred_inst_list = []

            #split long video into clips to form a batch input 
            if video_len > clip_length:
                num_clips = math.ceil(video_len/clip_length)
                logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
                for c in range(num_clips):
                    start_idx = c*clip_length
                    end_idx = (c+1)*clip_length
                    clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
                    clip_output, clip_instances = self.ytb_inference(clip_inputs)
                    logits_list.extend(clip_output['pred_logits'])  # [B, N, C]
                    boxes_list.extend(clip_output['pred_boxes'])
                    embed_list.extend(clip_output['pred_inst_embed'])
                    masks_list.extend(clip_output['pred_masks'])
                    pred_inst_list.extend(clip_instances)
                    # masks_list.append(clip_output['pred_masks'].to(self.merge_device))
                # 补一个 image_id 从数据流开始
                output = {
                    'pred_logits':logits_list,  # [video_len, N, C]
                    'pred_boxes':boxes_list,  # [ATTN] 因为尺寸可能不同，就不cat了 反正后面Inference只是for一遍
                    'pred_inst_embed':embed_list,
                    'pred_masks':masks_list,
                    'image_ids': batched_inputs[0]['image_ids']
                }    
            else:
                images = self.preprocess_image(batched_inputs)
                pass
            # save_for_recal_test -------------------
            # 这里就不存检测了 只看跟踪结果
            assert len(output['pred_boxes']) == video_len
            outdir = os.path.join(self.output_dir, 'props')
            imglike = batched_inputs[0]['file_names'][0]  # datasets/tao/frames/val/AVA/Di1MG6auDYo_scene_2_24458-25635/frame0101.jpg
            video_src = imglike.split('/')[4]
            video_name = imglike.split('/')[5]
            curr_outdir = os.path.join(outdir, video_src, video_name)  # [TODO] args.outdir
            if not os.path.exists(curr_outdir):
                os.makedirs(curr_outdir)
            results = pred_inst_list
            assert len(results) == video_len
            input_imgs = [{'height': batched_inputs[0]['height'], 'width': batched_inputs[0]['width']} for _ in range(video_len)]
            processed_results = OWTMASK._postprocess(results, input_imgs, images.image_sizes)
            # print('debug: infer, len(processed_results) = {}, len_results = {}, video_len = {}'.format(len(processed_results), len(results), video_len))
            for i, det_result in enumerate(processed_results):
                # print('i = {}, len(det_result[instances]) = {}'.format(i, len(det_result['instances'])))  # i = 16, len(det_result[instances]) = 1000
                valid_classes = []
                if self.roi_heads.num_classes == 1:
                    valid_classes = [0]
                else:
                    valid_classes = list(range(80))
                # print('valid_classes = {}'.format(valid_classes))
                store_TAOnpz(det_result, batched_inputs[0]['file_names'][i], valid_classes, curr_outdir)
            return 
            # -------------------
            # idol_tracker = IDOL_Tracker(
            #         # init_score_thr= 0.2,
            #         init_score_thr=0.0,
            #         obj_score_thr=0.1,
            #         nms_thr_pre=0.5, 
            #         nms_thr_post=0.05,
            #         # addnew_score_thr = 0.2,
            #         addnew_score_thr=0.0,
            #         # match_score_thr=0.5,
            #         memo_tracklet_frames = 10,
            #         memo_momentum = 0.8,
            #         long_match = self.inference_tw,
            #         frame_weight = (self.inference_tw|self.inference_fw),
            #         temporal_weight = self.inference_tw,
            #         memory_len = self.memory_len
            #         )
            idol_tracker = IDOL_Tracker(
                            # init_score_thr= 0.2,
                            init_score_thr=0.0,
                            obj_score_thr=0.1,  # 没用
                            nms_thr_pre=0.2, 
                            nms_thr_post=0.05,
                            # addnew_score_thr = 0.2,
                            addnew_score_thr=0.0,
                            # match_score_thr=0.5,
                            memo_tracklet_frames = 10,
                            memo_momentum = 0.8,
                            long_match = self.inference_tw,
                            frame_weight = (self.inference_tw|self.inference_fw),
                            temporal_weight = self.inference_tw,
                            memory_len = self.memory_len,
                            # match_metric='mix'
                            )
            height = batched_inputs[0]['height']  # 原始图片尺寸
            width = batched_inputs[0]['width']
            video_output = self.inference(output, idol_tracker, (height, width), images.image_sizes[0])
            # 

            return video_output

    def coco_inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        # print('batched_inputs = {}'.format(batched_inputs))
        # print('pre img.shape = {}'.format(batched_inputs[0]['image'].shape))
        # images = self.preprocess_image(batched_inputs)  # [ATTN]  不知道为何之前这个是可以运行的
        images = ImageList.from_tensors([batched_inputs[0]['image'].to(self.device)])  
        # print('images.tensor.shape = {}'.format(images.tensor.shape))
        features = self.backbone(images.tensor)
        # print('len_features = {}, keys = {}'.format(len(features), features.keys()))
        # print('debug: inference, images.image_sizes = {}'.format(images.image_sizes))
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            print('len_props = {}'.format(len(proposals)))  # [TODO] 为啥明明只有一层feat但是有3层框
            results, _ = self.roi_heads(images, features, proposals, None)  # [HERE]
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        print('debug: infer, do_postprocess = {}'.format(do_postprocess))
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return OWTMASK._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def ytb_inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)  # 因为images有好几张 所以proposals也有好几个
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        # [TODO] 这里应该是k太小了, 这个地方已经是超内存了
        # output= self.roi_heads(images, features, proposals, None)  # [TODO] 在roi里面补一个非常简短的过滤 例如nms0.9 top300
        # print('debug: video inference, output[pred_logits].shape = {}'.format(output['pred_logits'].shape))

        pred_instances_list = []
        clip_len = len(images)
        # print('clip_len = {}'.format(clip_len))
        for i in range(clip_len):
            clip_features = {}
            for k,v in features.items():
                clip_features[k] = v[i: i+1]
            pred_instances_list.extend(self.roi_heads(images, clip_features, proposals[i:i+1], None))  # [ATTN] imagelist类不能直接裁剪 但是roi_head里也没用到，这里也直接不裁剪了
            # print('len = {}'.format(len(pred_instances_list)))
        
        # pred_instances = Instances.cat(pred_instances_list)
        # [ATTN] 取消了下面的stack 因为rpn的过滤可能不足1000
        output = {
                    'pred_logits': [_.scores.view(-1, 1) for _ in pred_instances_list], # [B, N, C]  [ATTN] 给scores升维
                    'pred_boxes': [_.pred_boxes.tensor for _ in pred_instances_list],  # 这个是Boxes类
                    'pred_inst_embed': [_.embeds for _ in pred_instances_list],
                    'pred_masks': [_.pred_masks for _ in pred_instances_list],
                }
        # print('shape list = {}'.format([_.shape for _ in output.values()]))
        # shape list = [torch.Size([10, 300, 1]), torch.Size([10, 300, 4]), torch.Size([10, 300, 256]), torch.Size([10, 300, 1, 14, 14])]
        return output, pred_instances_list

    def inference(self, outputs, tracker, ori_size, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # results = []
        video_dict = {}
        # print('debug: video inference, outputs[pred_logits] = {}'.format(outputs['pred_logits'].shape))
        # video_logits = outputs['pred_logits'][:, :, :1]  # [video_len, N, C]
        video_logits = [_[:, :1] for _ in outputs['pred_logits']]
        video_output_masks = outputs['pred_masks']
        video_output_boxes = outputs['pred_boxes']
        video_output_embeds = outputs['pred_inst_embed']
        video_image_ids = outputs['image_ids']
        vid_len = len(video_logits)
        for i_frame, (logits, output_mask, output_boxes, output_embed) in enumerate(zip(
            video_logits, video_output_masks, video_output_boxes, video_output_embeds
         )):
            scores = logits.cpu().detach()  # 
            # print('scores.shape ={}, self.inference_select_thres = {}'.format(scores.shape, self.inference_select_thres))  # [1000, 1], 0.1
            max_score, _ = torch.max(logits,1)
            indices = torch.nonzero(max_score>self.inference_select_thres, as_tuple=False).squeeze(1)  # self.inference_select_thres = 0.1
            print('indices.long().sum() = {}'.format((max_score>self.inference_select_thres).long().sum()))  #  OWTMASK vid4 只有400+
            # [ATTN] 这个地方其实过滤了很多有价值的框 因为本身这个0.1的阈值就卡掉了近一半框
            # 两个方案
            # 1. 将这个阈值直接去掉，直接用后面比较严格的nms来决定，保留那些低分但是重合度不高的框
            # 2. 换指标例如obj + bg 但实际上props recall说明这个score已经够了
            if len(indices) == 0:
                topkv, indices_top1 = torch.topk(scores.max(1)[0],k=1)
                indices_top1 = indices_top1[torch.argmax(topkv)]
                indices = [indices_top1.tolist()]
            else:
                nms_scores,idxs = torch.max(logits[indices],1)
                boxes_before_nms = output_boxes[indices]
                keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)  # [TODO] 前置处理的时候的nms_thres要比这个高 否则就没意义了这一步
                indices = indices[keep_indices]
                print('len_indices = {}'.format(len(indices)))  # OWTMASK vid4还有300+
            box_score = torch.max(logits[indices],1)[0]  # 过滤过后的
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)  # [N>thr, 5]  这里将score拼到了box最后一维
            det_labels = torch.argmax(logits[indices],dim=1)  # [N>thr] 类别
            track_feats = output_embed[indices]
            det_masks = output_mask[indices]
            bboxes, labels, ids, indices = tracker.match(  # [ATTN]
            bboxes=det_bboxes,
            labels=det_labels,
            masks = det_masks,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            print('len_ids = {}'.format(len(ids)))  # 1 正常应该有40-50  OWTMASK vid4只有10+ 20+
            # 上面已经过滤了id=-1的
            for query_i, id in zip(indices,ids):  # 注意这个query_id仍对应原本output_mask的Index
                if id in video_dict.keys():
                    video_dict[id]['boxes'].append(output_boxes[query_i])  # {id: {'masks': []}}
                    video_dict[id]['scores'].append(scores[query_i])  # [C]
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1  # valid表示出现的次数
                else:
                    video_dict[id] = {
                        'boxes':[None for fi in range(i_frame)], 
                        'scores':[None for fi in range(i_frame)], 
                        'valid':0}  # 新建一个轨迹 之前没出现的补None
                    video_dict[id]['boxes'].append(output_boxes[query_i])
                    video_dict[id]['scores'].append(scores[query_i])
                    video_dict[id]['valid'] = video_dict[id]['valid'] + 1  # valid从1开始

            for k,v in video_dict.items():  # 有些在缓存中没匹配到的就 直接进行补充None
                if len(v['boxes'])<i_frame+1: #padding None for unmatched ID
                    v['scores'].append(None)
                    v['boxes'].append(None)


            #  filtering sequences that are too short in video_dict (noise)，the rule is: if the first two frames are None and valid is less than 3
            # if i_frame>8:
            #     del_list = []
            #     for k,v in video_dict.items():
            #         if v['boxes'][-1] is None and  v['boxes'][-2] is None and v['valid']<3:  # 如果连续两帧为None且出现次数<3直接删掉  [ATTN] 所以如果关联不好可能真的会删掉
            #             del_list.append(k)   
            #     for del_k in del_list:
            #         video_dict.pop(del_k)                      
        # 跟踪结束进行的后处理
        del outputs
        logits_list = []
        boxes_list = []
        # print('num_ids = {}'.format(len(video_dict)))
        for inst_id,m in  enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            scores_temporal = []
            for k in score_list_ori:
                if k is not None:
                    scores_temporal.append(k)  # 取出该轨迹中所有不为None的score
            logits_i = torch.stack(scores_temporal)  # [num_feiNone, C]
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)  # [C]
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            else:
                print('non valid temporal_score_type')
                import sys;sys.exit(0)
            # print('logits_i.shape = {}'.format(logits_i.shape))  # [C]
            logits_list.append(logits_i)
            # print('ori_h = {}, ori_w = {}'.format(ori_size[1], ori_size[1]))
            # category_id = np.argmax(logits_i.mean(0))
            boxes_list_i = []
            for n in range(vid_len):
                box_i = video_dict[m]['boxes'][n]
                if box_i is None:
                    boxes_list_i.append(None)
                else:
                    # print('box_i.shape = {}'.format(box_i.shape))  # [4]
                    pred_box_i = box_i
                    # print('pred_box_i.shape = {}'.format(pred_box_i.shape))  # [4]
                    # print('debug: ori_size = {}, image_sizes = {}'.format(ori_size, image_sizes)) 
                    # debug: ori_size = (480, 640), image_sizes = (480, 640)
                    # print('pred_box_i = {}'.format(pred_box_i))
                    # pred_box_i = tensor([630.3299, 196.0527, 638.8394, 281.7104], device='cuda:0')
                    pred_box_i[0::2] *= ori_size[1] / image_sizes[1] # w
                    pred_box_i[1::2] *= ori_size[0] / image_sizes[0] # h
                    boxes_list_i.append(pred_box_i)
            boxes_list.append(boxes_list_i)
        if len(logits_list)>0:
            pred_cls = torch.stack(logits_list)  # [num_ids, C] 将所有轨迹的Cls score求了个均值
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:  # True
                # print('pred_cls.shape = {}'.format(pred_cls.shape))  # [num_ids, C]
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)  # idol = 0.05
                scores = pred_cls[is_above_thres]  # [num_meet_above]  对应满足条件的score 但同一个id可能出现多个cls满足
                labels = is_above_thres[1]  # [num_meet_above] 对应类别idx
                boxes_list_mc = [] # masks_list multi_cls
                for idx in is_above_thres[0]:
                    boxes_list_mc.append(boxes_list[idx])
                out_boxes = boxes_list_mc  # list[ [tenor(4), None, ...]]
            else:
                scores, labels = pred_cls.max(-1)
                out_boxes = boxes_list
            
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_boxes = []
        video_output = {
            "image_size": ori_size,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_boxes": out_boxes,
            "image_ids": video_image_ids
        }

        return video_output

    def preprocess_image(self, batched_inputs):  # coco inference不能使用下面这个
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:  # 同一张图的两个增强view
                # print('debug: train, frame.shape = {}'.format(frame.shape))
                # debug: train, frame.shape = torch.Size([640, 1138]) 这里有个bug 如果是在coco inference里面只有一张图 那么就会按通道维度for出来
                images.append(self.normalizer(frame.to(self.device)))
        # print('len_images = {}'.format(len(images)))
        # images = ImageList.from_tensors(images[2:3])  # 
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)  # 如果不加 self.backbone.size_divisibility 那么在fpn里面会尺寸报错
        return images
    

    def prepare_targets(self, targets):
        new_instances = []
        for bid,targets_per_image in enumerate(targets):
            # new_instance = Instances(targets_per_image.image_size)
            valid_id = targets_per_image.gt_ids != -1
            new_instance = targets_per_image[valid_id]  # 使用getitem方法  这里我跟idol不同，我这里是key 和 ref两种的inst为空的情况都去掉了 所以两张图都可以一起训练检测，而idol里面是只用key的gt_id过滤，所以只有key训练检测
            new_instances.append(new_instance)  # 在roi head里面再去解耦开key 和ref
        return new_instances

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            # print('debug: inference, input_per_image[height] = {}, input_per_image[width] = {}, image_size = {}'.format(input_per_image['height'], input_per_image['width'], image_size[:2]))
            # debug: inference, input_per_image[height] = 720, input_per_image[width] = 1280, image_size = (640, 1138)
            # print('debug: infer, results_per_image.image_size = {}'.format(results_per_image.image_size))
            # debug: infer, results_per_image.image_size = (640, 1138)
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

