# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Callable, List, Union
import torch
from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .panoptic_target_generator import PanopticTargetGenerator
from .instance_target_generator import InstanceTargetGenerator

__all__ = ["PostDatasetMapper"]


class PostDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name"
    2. Applies random scale, crop and flip transforms to image and label
    3. Prepare data to Tensor and generate training targets from label
    """

    @configurable
    def __init__(
        self,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        instance_mask_format: str = "bitmask",
        panoptic_target_generator: Callable,
        instance_target_generator: Callable,
        test: bool = False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.            
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
            instance_target_generator: a callable that takes "instances"
                to generate training targets for the model.
            test: whether to load val/test dataset, no targets will be applied, batchsize == 1
        """
        # fmt: off
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.test = test
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

        self.panoptic_target_generator = panoptic_target_generator
        self.instance_target_generator = instance_target_generator

    @classmethod
    def from_config(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        instance_target_generator = InstanceTargetGenerator(
            ignore_label=meta.ignore_label,
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
        )

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "panoptic_target_generator": panoptic_target_generator,
            "instance_target_generator": instance_target_generator,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        if dataset_dict["prev_file_name"]:
            prev_image = utils.read_image(dataset_dict["prev_file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Don't generate targets for test
        if self.test:
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if dataset_dict["prev_file_name"]:
                dataset_dict["prev_image"] = torch.as_tensor(np.ascontiguousarray(prev_image.transpose(2, 0, 1)))
            return dataset_dict
            
        # Panoptic label is encoded in RGB image.
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
        else:
            pan_seg_gt = None

        # Reuses semantic transform for panoptic labels.
        # todo...
        aug_input = T.AugInput(image, prev_image=prev_image, sem_seg=pan_seg_gt)
        transforms = self.augmentations(aug_input)
        image, prev_image, pan_seg_gt = aug_input.image, aug_input.prev_image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["prev_image"] = torch.as_tensor(np.ascontiguousarray(prev_image.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            dataset_dict["annotations"] = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                )
                for obj in dataset_dict["annotations"]
            ]
        
        if "prev_annotations" in dataset_dict:
            dataset_dict["prev_annotations"] = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                )
                for obj in dataset_dict.pop("prev_annotations")
            ]

        if pan_seg_gt is not None:
            # Generates training targets for Panoptic-DeepLab.
            targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
            dataset_dict.update(targets)
        else:
            targets = self.instance_target_generator(dataset_dict["annotations"], dataset_dict["prev_annotations"])
            dataset_dict.update(targets)

        return dataset_dict
