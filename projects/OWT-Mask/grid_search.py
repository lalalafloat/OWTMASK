import os
import torch
import json
import copy
import numpy as np
import torchvision.ops as ops
from tqdm import tqdm
from owtmask.models.tracker import IDOL_Tracker
from owtmask.data.seqstao_eval import instances_to_coco_json_video
# 考虑到score的召回已经很好，而且我们不需要过滤掉低分框，因为unknown可能分数低，但是还是被召回了
# 所以仅做一次box_nms，其余过滤条件全部置零，另外轨迹长度过滤可以搜索一下
device = torch.device('cuda:0')
class Offline_Tracker():
    def __init__(self, boxnms, valid_num) -> None:
        self.boxnms = boxnms
        self.valid_num = valid_num
        self.inference_tw = True
        self.inference_fw = True
        self.inference_select_thres = 0.0
        self.memory_len = 3  # 因为这个玩意是3 所以proposals之前的长度才为3
        self.temporal_score_type = 'mean'
        self.is_multi_cls = True
        self.apply_cls_thres = 0.05
        # self.apply_cls_thres = 0.00005  # obj用0.0
        self._output_dir = 'grid_search_results_80cls_new/score_boxnms{}_validnum{}'.format(boxnms, valid_num)  # [TODO]
        print('-----start: {} -----'.format(self._output_dir))
        
    def track(self, ):
        # props_dir = 'TEST_rpn1k_ytb_inference_prop_save/props' 
        # props_dir = 'TEST_rpn1k_ytb_80cls_debug_inference_prop_save/props'
        props_dir = 'TEST_rpn1k_ytb_80cls_debug_valid_classes_inference_prop_save/props'
        val_json = 'datasets/tao/annotations/validation_agn.json'
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        # print(val_data.keys())  # dict_keys(['videos', 'annotations', 'tracks', 'images', 'info', 'categories', 'licenses'])
        print(val_data['videos'][0])
        print(val_data['images'][0])
        # return
        vinfo = {}
        for video in val_data['videos']:
            vid = video['id']
            vinfo[vid] = video
            vinfo[vid]['frames'] = []
        for img in val_data['images']:
            vid = img['video_id']
            vinfo[vid]['frames'].append(img)
        for k,v in vinfo.items():
            v['frames'] = sorted(v['frames'], key=lambda x: x['frame_index'])
        # print([_['frame_index'] for _ in list(vinfo.values())[0]['frames']])

        for vid, video in tqdm(vinfo.items()):
            logits_list, boxes_list, embed_list, masks_list = [], [], [], []  # B x [N, C]
            image_ids = []
            for img in video['frames']:  # 本身排过序了
                image_ids.append(img['id'])
                img_name = img['file_name'].split('/')[1:]
                img_name = '/'.join(img_name).replace('.jpg', '.npz').replace('.png', '.npz')
                res_npz = os.path.join(props_dir, img_name)
                if os.path.exists(res_npz):
                    
                    proposals = np.load(res_npz, allow_pickle=True)['arr_0'].tolist()
                    # print('len_proposals = {}'.format(len(proposals)))  # cls_agn = 1000  80类的时候这里不是1K
                    # print(proposals[0].keys())  # dict_keys(['category_id', 'bbox', 'score', 'bg_score', 'objectness', 'embeddings'])
                    logits, boxes, embed, masks = [], [], [], []
                    for prop in proposals:
                        # print('prop[score] = {}'.format(prop['score']))
                        # print('load prop[score].shape = {}'.format(prop['score'].shape))
                        logits.append(prop['score'])
                        # logits.append(prop['objectness'])
                        boxes.append(prop['bbox'])
                        embed.append(prop['embeddings'])
                    # logits = torch.Tensor(logits).unsqueeze(-1).to(device)  # [TODO] device
                    # boxes = torch.Tensor(boxes).to(device)
                    # embed = torch.Tensor(embed).to(device) / 1e4
                    logits = torch.Tensor(logits).unsqueeze(-1)  # [TODO] device  好像测下来不用gpu反而快
                    boxes = torch.Tensor(boxes)
                    embed = torch.Tensor(embed) / 1e4
                logits_list.append(logits)
                boxes_list.append(boxes)
                embed_list.append(embed)
                masks_list.append(None)
            output = {
                        'pred_logits':logits_list,  # [video_len, N, C]
                        'pred_boxes':boxes_list,  # [ATTN] 因为尺寸可能不同，就不cat了 反正后面Inference只是for一遍
                        'pred_inst_embed':embed_list,
                        'pred_masks':masks_list,
                        'image_ids': image_ids
                    }    


            # 初始化tracker
            idol_tracker = IDOL_Tracker(
                            # init_score_thr= 0.2,
                            init_score_thr=0.0,
                            obj_score_thr=0.1,  # 没用
                            nms_thr_pre=self.boxnms, 
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
            height = video['height']
            width = video['width']
            video_output = self.inference(output, idol_tracker, (height, width), (height, width))

            # video_input 格式参考 projects/OWT-Mask/owtmask/data/seqstao_eval.py line208
            video_input = [
                {
                    'video_id': vid,
                    'length': len(image_ids)
                }
            ]
            self.process(video_input, video_output)

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
        print('debug: search inference, oututs[pred_logits][0].shape = {}'.format(outputs['pred_logits'][0].shape))
        # debug: search inference, oututs[pred_logits][0].shape = torch.Size([96, 1])  为啥不是1K  原来的是1K的
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
            # values, indices = torch.topk(max_score, k=300)  # [TODO] 和上面互相替换
            # print('indices = {}'.format(indices))
            # print('indices.long().sum() = {}'.format((max_score>self.inference_select_thres).long().sum()))  #  OWTMASK vid4 只有400+
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
                # print('output_boxes.shape = {}, scores.shape = {}'.format(output_boxes.shape, scores.shape))
                boxes_before_nms = output_boxes[indices]
                # keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.9)  # [TODO] 前置处理的时候的nms_thres要比这个高 否则就没意义了这一步
                # keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,1.0)  # [TODO] 前置处理的时候的nms_thres要比这个高 否则就没意义了这一步
                # indices = indices[keep_indices]
                indices = indices
                # print('len_indices = {}'.format(len(indices)))  # OWTMASK vid4还有300+
            box_score = torch.max(logits[indices],1)[0]  # 过滤过后的
            det_bboxes = torch.cat([output_boxes[indices],box_score.unsqueeze(1)],dim=1)  # [N>thr, 5]  这里将score拼到了box最后一维
            det_labels = torch.argmax(logits[indices],dim=1)  # [N>thr] 类别
            track_feats = output_embed[indices]
            # det_masks = output_mask[indices]
            bboxes, labels, ids, indices = tracker.match(  # [ATTN]
            bboxes=det_bboxes,
            labels=det_labels,
            # masks = det_masks,
            masks=None,
            track_feats=track_feats,
            frame_id=i_frame,
            indices = indices)
            indices = torch.tensor(indices)[ids>-1].tolist()
            ids = ids[ids > -1]
            ids = ids.tolist()
            # print('len_ids = {}'.format(len(ids)))  # 1 正常应该有40-50  OWTMASK vid4只有10+ 20+
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
            if i_frame>8:
                del_list = []
                for k,v in video_dict.items():
                    if v['boxes'][-1] is None and  v['boxes'][-2] is None and v['valid']<self.valid_num:  # 如果连续两帧为None且出现次数<3直接删掉  [ATTN] 所以如果关联不好可能真的会删掉
                        del_list.append(k)   
                for del_k in del_list:
                    video_dict.pop(del_k)                      
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
        # print("len_pred_cls = {}".format(len(pred_cls)))
        if len(pred_cls) > 0:
            if self.is_multi_cls:  # True
                # print('pred_cls.shape = {}'.format(pred_cls.shape))  # [num_ids, C]
                is_above_thres = torch.where(pred_cls > self.apply_cls_thres)  # idol = 0.05
                scores = pred_cls[is_above_thres]  # [num_meet_above]  对应满足条件的score 但同一个id可能出现多个cls满足
                # print('after scores.shape = {}'.format(scores.shape))
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

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # print('dist = {}'.format(self._distributed))
        prediction = instances_to_coco_json_video(inputs, outputs)  # [ATTN]
        # 过程中存储
        video_id = inputs[0]["video_id"]
        sub_dir = 'tao'
        out_dir = os.path.join(self._output_dir, sub_dir)  # [TODO]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "vid_{}.json".format(video_id))
        with open(file_path, "w") as f:
            f.write(json.dumps(prediction))


if __name__ == '__main__':
    # nms_thr = 0.15
    # while nms_thr < 0.65:
    #     nms_thr += 0.05
    #     for valid in range(0, 4, 1):
    #         offline_tracker = Offline_Tracker(nms_thr, valid)
    #         offline_tracker.track()
    offline_tracker = Offline_Tracker(0.25, 2)
    offline_tracker.track()
    pass