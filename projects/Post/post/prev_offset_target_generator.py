# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

class PrevOffsetTargetGenerator:
    """
    Generates previous offset training targets for Post.
    """

    def __init__(
        self,
        ignore_label,
        sigma=8,
    ):
        """
        Args:
            ignore_label: Integer, the ignore label for semantic segmentation.
            sigma: the sigma for Gaussian kernel.
        """
        self.ignore_label = ignore_label

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, annotations, prev_annotations):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18  # noqa

        Args:
            annotations: annotations of instances in one frame
            prev_annotations: annotations of the previous frame, 
                that are also present in the current frame

        Returns:
            A dictionary with fields:
                - sem_seg: Tensor, semantic label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(H, W).
                - center_points: List, center coordinates, with tuple
                    (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - prev_offset: Tensor, offset, shape=(2, H, W), first dim is
                    (offset_y, offset_x).
                - center_weights: Tensor, ignore region of center prediction,
                    shape=(H, W), used as weights for center regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
                - prev_offset_weights: Tensor, ignore region of the previous offset prediction,
                    shape=(H, W), used as weights for offset regression 0 is
                    ignore, 1 is has instance. Multiply this mask to loss.
        """
        image_size = annotations[0]["segmentation"].shape
        (height, width) = image_size
        center = np.zeros(image_size, dtype=np.float32)
        center_pts = []
        prev_offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )

        # joint_masks of all annotations, to generate center and offset weights
        prev_joint_mask = None
    
        for anno in annotations:

            mask = anno["segmentation"]

            # mask index x and y of where the mask is 'true'
            mask_index = np.where(mask)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])
            if np.isnan(center_y) or np.isnan(center_x):
                continue

            # get previous mask if there is one (same object id)   
            prev_mask = [
                prev_anno["segmentation"] 
                for prev_anno in prev_annotations 
                if prev_anno["object_id"] == anno["object_id"]
                ]

            # generate previous joint mask
            if np.any(prev_mask):
                prev_mask = prev_mask[0]
                if prev_joint_mask is None:
                    prev_joint_mask = prev_mask
                else:
                    prev_joint_mask = np.logical_or(prev_joint_mask, mask)
            else:
                continue

            # generate previous offset (2, h, w) -> (y-dir, x-dir) i
            mask_index = np.where(prev_mask)
            prev_offset[0][mask_index] = center_y - y_coord[mask_index]
            prev_offset[1][mask_index] = center_x - x_coord[mask_index]

        if np.any(prev_joint_mask):
            prev_offset_weights = np.where(prev_joint_mask, 1, 0)
        else:
            prev_offset_weights = np.zeros(image_size, dtype=np.uint8)
        # Using None as is equivalent to using numpy.newaxis
        prev_offset_weights = prev_offset_weights[None]
        return dict(
            prev_offset=torch.as_tensor(prev_offset.astype(np.float32)),
            prev_offset_weights=torch.as_tensor(prev_offset_weights.astype(np.float32)),
        )
