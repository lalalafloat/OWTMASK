# ------------------------------------------------------------------------
# IDOL: In Defense of Online Models for Video Instance Segmentation
# Copyright (c) 2022 ByteDance. All Rights Reserved.


import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["COCO_CLIP_DatasetMapper"]

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class COCO_CLIP_DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None
        
        self.same_crop =  cfg.INPUT.PRETRAIN_SAME_CROP

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):  # only used for training
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # print('dataset_dict in mapper = {}'.format(dataset_dict))
        # dataset_dict in mapper = {'file_name': 'datasets/./tao/frames/train/Charades/4H61U/frame0110.jpg', 'height': 720, 'width': 1280, 'image_id': 1524486, 'annotations': [{'iscrowd': 0, 'bbox': [209, 148, 189, 536], 'category_id': 0, 'longscore': 0.8300819993019104, 'segmentation': [[209, 148, 398, 148, 398, 684, 209, 684]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [432, 446, 279, 266], 'category_id': 0, 'longscore': 0.680676281452179, 'segmentation': [[432, 446, 711, 446, 711, 712, 432, 712]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [4, 661, 236, 57], 'category_id': 0, 'longscore': 0.5973333716392517, 'segmentation': [[4, 661, 240, 661, 240, 718, 4, 718]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [183, 501, 116, 216], 'category_id': 0, 'longscore': 0.4537712633609772, 'segmentation': [[183, 501, 299, 501, 299, 717, 183, 717]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [1, 505, 144, 148], 'category_id': 0, 'longscore': 0.3760734796524048, 'segmentation': [[1, 505, 145, 505, 145, 653, 1, 653]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [830, 439, 56, 69], 'category_id': 0, 'longscore': 0.2500464618206024, 'segmentation': [[830, 439, 886, 439, 886, 508, 830, 508]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [866, 371, 21, 74], 'category_id': 0, 'longscore': 0.22115252912044525, 'segmentation': [[866, 371, 887, 371, 887, 445, 866, 445]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [345, 174, 41, 131], 'category_id': 0, 'longscore': 0.2198409140110016, 'segmentation': [[345, 174, 386, 174, 386, 305, 345, 305]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [363, 166, 44, 465], 'category_id': 0, 'longscore': 0.21333521604537964, 'segmentation': [[363, 166, 407, 166, 407, 631, 363, 631]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}, {'iscrowd': 0, 'bbox': [374, 242, 31, 95], 'category_id': 0, 'longscore': 0.2091483175754547, 'segmentation': [[374, 242, 405, 242, 405, 337, 374, 337]], 'bbox_mode': <BoxMode.XYWH_ABS: 1>}]}
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if not self.same_crop: #crop twice independently
            #### process key frame
            # dataset_dict_ref = copy.deepcopy(dataset_dict)
            image_ref = copy.deepcopy(image)

            if self.crop_gen is None:
                image_key, transforms_key = T.apply_transform_gens(self.tfm_gens, image)
            else:
                if np.random.rand() > 0.5:
                    image_key, transforms_key = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image_key, transforms_key = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                    )

            key_image_shape = image_key.shape[:2]  # h, w
            dataset_dict["image"] = []
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image_key.transpose(2, 0, 1))))

            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
            key_annotations = dataset_dict.pop("annotations")
            ref_annotations = copy.deepcopy(key_annotations)
            
            # 这里面只对annos的box, seg, keypoint做了变换,  
            annos_key = [
                utils.transform_instance_annotations(obj, transforms_key, key_image_shape)
                for obj in key_annotations
                if obj.get("iscrowd", 0) == 0
            ]
            
            # print('annos_key = {}'.format(annos_key))
            instances_key = utils.annotations_to_instances(annos_key, key_image_shape, mask_format="bitmask")
            
            
            #### process reference frame ##########

            if self.crop_gen is None:
                image_ref, transforms_ref = T.apply_transform_gens(self.tfm_gens, image_ref)
            else:
                if np.random.rand() > 0.5:
                    image_ref, transforms_ref = T.apply_transform_gens(self.tfm_gens, image_ref)
                else:
                    image_ref, transforms_ref = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image_ref
                    )

            ref_image_shape = image_ref.shape[:2]
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image_ref.transpose(2, 0, 1))))
            annos_ref = [
                utils.transform_instance_annotations(obj, transforms_ref, ref_image_shape)
                for obj in ref_annotations
                if obj.get("iscrowd", 0) == 0
            ]
            instances_ref = utils.annotations_to_instances(annos_ref, ref_image_shape, mask_format="bitmask")


            _gt_ids = list(range(1,1+len(annos_ref)))
            instances_key.gt_ids = torch.tensor(_gt_ids)
            instances_ref.gt_ids = torch.tensor(_gt_ids)
            dataset_dict["instances"] = [filter_empty_instances(instances_key),  filter_empty_instances(instances_ref)]
            # for key/ref frame， we don't remove empty instances，but mark them with gt_ids=-1, and process them in idol.py
            # gt_ids has no practical meaning, we just use it as a flag to indicate whether an instance exists, 
            # idx indicates the object correspondence between key&reference frame
            
            return dataset_dict


        else:
            if self.crop_gen is None:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                if np.random.rand() > 0.5:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                    )

            image_shape = image.shape[:2]  # h, w

            key_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            ref_image = copy.deepcopy(key_image)
            dataset_dict["image"] = [key_image,ref_image]

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.mask_on:
                        anno.pop("segmentation", None)
                    anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")

                _gt_ids = list(range(1,1+len(annos)))
                instances.gt_ids = torch.tensor(_gt_ids)

                dataset_dict["instances"] = [filter_empty_instances(instances), filter_empty_instances(instances)]
            return dataset_dict