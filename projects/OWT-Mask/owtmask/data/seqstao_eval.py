# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import multiprocessing
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
# from pycocotools.ytvos import YTVOS
from .datasets.seqstao import TAO
import torch.distributed as dist

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager


class SEQSTAOEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._ytvis_api = TAO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def reset(self):
        self._predictions = []

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
        
        # self._predictions.extend(prediction)  # 返回个空

    def evaluate(self):
        # 改成存储为tao格式的json
        # 整体存内存不够 10+个视频的Json就会卡
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:  # [ATTN] 下面就是多进程同步
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        # 给proposal_id统一赋值
        for idx,anno in enumerate(predictions):
            anno['id'] = idx + 1

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:  # [ATTN] 按照TAO的方式存储
            # 内存不够
            pass
            # PathManager.mkdirs(self._output_dir)
            # file_path = os.path.join(self._output_dir, "test.json")
            # with open(file_path, "w") as f:
            #     f.write(json.dumps(predictions))

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in predictions:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]  # [ATTN] 用不上，TAO里面默认就是0-79

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        self._logger.info("Annotations are not available for evaluation.")
        return


def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    video_length = inputs[0]["length"]
    # return []  # 先生成props 下面不管了 [TODO]  ytb测试的适合需要打开
    scores = outputs["pred_scores"]  # [num_tids]
    labels = outputs["pred_labels"]
    boxes = outputs["pred_boxes"]  # [ [Tensor or None]]
    image_ids = outputs['image_ids']
    assert len(boxes[0]) == len(image_ids), "len_boxes[0] = {}, len_image_ids = {}".format(len(boxes[0]), len(image_ids))

    ytvis_results = []
    # 此处直接写成tao需要的json中的dict格式
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, boxes)):
        
        assert len(image_ids) == len(m)
        for (box, image_id) in zip(m, image_ids):
            if box is None:
                continue
            res = {}
            x1, y1, x2, y2 = box.long().tolist()
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            x1 = min(x1, x2)
            y1 = min(y1, y2)
            res = {
                'bbox': [x1, y1, w, h],
                'track_id': instance_id,
                'category_id': l,
                'image_id': image_id,
                'id': -1, # SEQSTAOEvaluator.curr_props_id(),  # [TODO] 没法做到进程同步 统一赋值一遍 甚至Track测试不需要prop['id']
                'video_id': video_id,
                'score': s
            }
            ytvis_results.append(res)

    return ytvis_results
