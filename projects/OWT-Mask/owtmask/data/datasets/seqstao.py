# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import json
import logging
from re import L
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_seqstao_json", "register_seqstao_instances"]

from pycocotools.ytvos import YTVOS
from collections import defaultdict
class TAO(YTVOS):
    # 改写一些函数以支持tao的测试
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, vids = {}, {}, {}
        vidToAnns,catToVids = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                vidToAnns[ann['video_id']].append(ann)
                anns[ann['id']] = ann

            
        if 'videos' in self.dataset:
            for vid in self.dataset['videos']:
                vids[vid['id']] = vid
        # 补充length 和 file_names 和 img_ids
        vid2imgs = defaultdict(list)
        for img_info in self.dataset['images']:
            vid2imgs[img_info['video_id']].append(img_info)
        for k,v in vid2imgs.items():
            vids[k]['length'] = len(v)
            v.sort(key=lambda x: x['frame_index'])
            vids[k]['file_names'] = [img_info['file_name'] for img_info in v]
            vids[k]['image_ids'] = [img_info['id'] for img_info in v]
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToVids[ann['category_id']].append(ann['video_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.vidToAnns = vidToAnns
        self.catToVids = catToVids
        self.vids = vids
        self.cats = cats


def load_seqstao_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = TAO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        # print('meta = {}'.format(meta))
        # Metadata(evaluator_type='ytvis', image_root='datasets/tao/frames', json_file='datasets/tao/annotations/tao_train_coco_1.json', name='ytvis_tao_train', thing_classes=['coco'], thing_colors=[[220, 20, 60]], thing_dataset_id_to_contiguous_id={1: 0})
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)  #[:5]  # [TEST] 加[:5] 进行少量测试
    # print(vids[0]) 原来的loadVids
    # {'id': 0, 'width': 640, 'height': 480, 'neg_category_ids': [342, 57, 651, 357, 738], 'not_exhaustive_category_ids': [805, 95], 'name': 'train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e', 'metadata': {'dataset': 'YFCC100M', 'user_id': '22634709@N00', 'username': 'Amsterdamized'}}

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["image_ids"] = vid_dict['image_ids']
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_seqstao_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_seqstao_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="seqstao", **metadata
    )


# 下面是测试的 没啥用
if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
