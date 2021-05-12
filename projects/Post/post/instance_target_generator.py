# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

class InstanceTargetGenerator(object):
    """
    Generates instance training targets for Post.
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

    def __call__(self, instances):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18  # noqa

        Args:
            instances: 

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
        image_size = instances.image_size
        # TODO: what to do with the don't care region
        instances = instances[instances.gt_classes != self.ignore_label]
        (height, width) = image_size
        masks = instances.gt_masks.tensor.numpy()
        center = np.zeros(image_size, dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        prev_offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )

        # joint_masks of all instances, to generate center and offset weights
        joint_mask = None
    
        for mask in masks:

            if joint_mask is None:
                joint_mask = mask
                continue
            joint_mask = np.logical_or(joint_mask, mask)

            # mask index x and y of where the mask is 'true'
            mask_index = np.where(mask)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])

            # generate center heatmap
            y, x = int(round(center_y)), int(round(center_x))
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            # start and end indices in default Gaussian image
            gaussian_x0, gaussian_x1 = max(0, -ul[0]), min(br[0], width) - ul[0]
            gaussian_y0, gaussian_y1 = max(0, -ul[1]), min(br[1], height) - ul[1]

            # start and end indices in center heatmap image
            center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
            center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
            center[center_y0:center_y1, center_x0:center_x1] = np.maximum(
                center[center_y0:center_y1, center_x0:center_x1],
                self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1],
            )

            # generate offset (2, h, w) -> (y-dir, x-dir)
            offset[0][mask_index] = center_y - y_coord[mask_index]
            offset[1][mask_index] = center_x - x_coord[mask_index]

            # TODO: generate previous offset (2, h, w) -> (y-dir, x-dir)
            prev_offset[0][mask_index] = center_y - y_coord[mask_index]
            prev_offset[1][mask_index] = center_x - x_coord[mask_index]

        # TODO: implement for previous offset
        prev_offset_weights = np.zeros(image_size, dtype=np.uint8)
        # set 1 where there is some mask and 0 else where
        center_weights = np.where(joint_mask, 1, 0)
        offset_weights = np.where(joint_mask, 1, 0)
        # Using None as is equivalent to using numpy.newaxis
        center_weights = center_weights[None]
        offset_weights = offset_weights[None]
        return dict(
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            prev_offset=torch.as_tensor(prev_offset.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
            prev_offset_weights=torch.as_tensor(prev_offset_weights.astype(np.float32)),
        )
