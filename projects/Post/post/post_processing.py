# Copyright (c) Facebook, Inc. and its affiliates.
# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py  # noqa

from collections import Counter
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import BitMasks, Instances
from .tracking import IouTracking
import torch
import numpy as np
import torch.nn.functional as F

class _PrevImage:
    """
    Used to store data about the segmentation in the previous image
    to generate a new panoptic image of the previous frame with current centers
    """
    __slots__ = ["sem_seg_result", "sem_seg", "thing_seg", "panoptic"]
    def __init__(self, sem_seg_result=None, sem_seg=None, thing_seg=None, panoptic=None):
        # [C, H, W]
        self.sem_seg_result = sem_seg_result
        # [1, H, W]
        self.sem_seg = sem_seg
        self.thing_seg = thing_seg
        self.panoptic = panoptic
class PostProcessing:
    """
    Post-processor for panoptic segmentation and tracking.
    """
    def __init__(
        self, 
        thing_ids, 
        label_divisor, 
        stuff_area, 
        void_label, 
        predict_instances,
        threshold=0.1, 
        nms_kernel=7,
        top_k=200,
        ):
        """
        Args:
            thing_ids: A set of ids from contiguous category ids belonging
                to thing categories.
            label_divisor: An integer, used to convert panoptic id =
                semantic id * label_divisor + s.
            stuff_area: An integer, remove stuff whose area is less tan stuff_area.
            void_label: An integer, indicates the region has no confident prediction.
            predict_instances: A bool whether to predict instances
            threshold: A float, threshold applied to center heatmap score.
            nms_kernel: An integer, NMS max pooling kernel size.
            top_k: An integer, top k centers to keep.
        """
        self.thing_ids = thing_ids
        self.label_divisor = label_divisor
        self.stuff_area = stuff_area
        self.void_label = void_label
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.top_k = top_k
        self.predict_instances = predict_instances
        self.prev_image = _PrevImage()
        self.tracking = IouTracking()
        self.height = None
        self.width = None

    def __call__(self, sem_seg_result, center, offset, prev_offset, input, output_size):
        """
        Post-processing for panoptic segmentation.
        Args:
            sem_seg: A Tensor of shape [1, H, W] of predicted semantic label.
            center: A Tensor of shape [1, H, W] of raw center heatmap output.
            offset: A Tensor of shape [2, H, W] of raw offset output. The order of
                second dim is (offset_y, offset_x).
            prev_offset: A Tensor of shape [2, H, W] of raw previous offset output. The order of
                second dim is (offset_y, offset_x).
            input: input image used to get the input height and width
            output_size: size of the output
        Returns:
            panoptic: A Tensor of shape [1, H, W], int64.
            center_points: A Tensor of shape [1, K, 2] where K is the number of center points.
            The order of second dim is (y, x).
            processed_result: A dict containing panoptic_seg, sem_seg and instances
        """
        processed_result = {}
        # input height and width usually !=  output_size
        (self.height, self.width) = (input.get("height"), input.get("width"))
        sem_seg_result = sem_seg_postprocess(sem_seg_result, output_size, self.height, self.width)
        center = sem_seg_postprocess(center, output_size, self.height, self.width)
        offset = sem_seg_postprocess(offset, output_size, self.height, self.width)
        if prev_offset is not None:
            prev_offset = sem_seg_postprocess(prev_offset, output_size, self.height, self.width)
        else:
            # reset ids and old instances for a new sequence
            self.tracking.reset()
        # For semantic segmentation evaluation.
        processed_result['sem_seg'] = sem_seg_result
        # Post-processing to get panoptic segmentation.
        panoptic, prev_panoptic = self.get_panoptic_segmentation(
            sem_seg_result.argmax(dim=0, keepdim=True),
            center,
            offset,
            prev_offset
        )
        # For panoptic segmentation evaluation.
        processed_result['panoptic_seg'] = (panoptic, None)
  
        if not self.predict_instances:
            return processed_result
        
        # For instance segmentation evaluation
        instances = self.get_instances(panoptic, center, sem_seg_result)
        if prev_panoptic is not None:
            prev_instances = self.get_instances(prev_panoptic, center, self.prev_image.sem_seg_result)
        else:
            prev_instances = None
        if instances:
            # assign object id to instances
            instances = self.tracking(instances, prev_instances)
            processed_result["instances"] = instances

        # Update semantic seg result
        self.prev_image.sem_seg_result = sem_seg_result
        return processed_result


    def get_panoptic_segmentation(self, sem_seg, center, offset, prev_offset):
        if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
            raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))
        if center.dim() != 3:
            raise ValueError(
                "Center prediction with un-supported dimension: {}.".format(center.dim())
            )
        if offset.dim() != 3:
            raise ValueError("Offset prediction with un-supported dimension: {}.".format(offset.dim()))

        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in list(self.thing_ids):
            thing_seg[sem_seg == thing_class] = 1

        center_points = self.get_center_points(center)
        ins_seg = self.get_instance_segmentation(sem_seg, center_points, offset, thing_seg)
        panoptic = self.merge_semantic_and_instance(sem_seg, ins_seg, thing_seg)
        if self.prev_image.thing_seg is not None and prev_offset is not None:
            prev_ins_seg = self.get_instance_segmentation(self.prev_image.sem_seg, center_points, prev_offset, self.prev_image.thing_seg)
            prev_panoptic = self.merge_semantic_and_instance(self.prev_image.sem_seg, prev_ins_seg, self.prev_image.thing_seg)
            prev_panoptic = prev_panoptic.squeeze(0)
        else:
            prev_panoptic = None

        # Update previous instance
        self.prev_image.sem_seg = sem_seg
        self.prev_image.thing_seg = thing_seg
        self.prev_image.panoptic = panoptic
        return panoptic.squeeze(0), prev_panoptic
    
    def get_center_points(self, center):
        """
        Find the center points from the center heatmap.
        Args:
            center: A Tensor of shape [1, H, W] of raw center heatmap output.
        Returns:
            A Tensor of shape [K, 2] where K is the number of center points. The
                order of second dim is (y, x).
        """
        # Thresholding, setting values below threshold to -1.
        center = F.threshold(center, self.threshold, -1)

        # NMS
        nms_padding = (self.nms_kernel - 1) // 2
        center_max_pooled = F.max_pool2d(
            center, kernel_size=self.nms_kernel, stride=1, padding=nms_padding
        )
        center[center != center_max_pooled] = -1

        # Squeeze first two dimensions.
        center = center.squeeze()
        assert len(center.size()) == 2, "Something is wrong with center heatmap dimension."

        # Find non-zero elements.
        if self.top_k is None:
            return torch.nonzero(center > 0)
        else:
            # find top k centers.
            top_k_scores, _ = torch.topk(torch.flatten(center), self.top_k)
            return torch.nonzero(center > top_k_scores[-1].clamp_(min=0))

    def get_instance_segmentation(self, sem_seg, center_points, offset, thing_seg):
        """
        Post-processing for instance segmentation, gets class agnostic instance id.
        Args:
            sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
            center_points: A Tensor of shape [K, 2] where K is the number of center points.
                The order of second dim is (y, x).
            offset: A Tensor of shape [2, H, W] of raw offset output. The order of
                second dim is (offset_y, offset_x).
            thing_seg: A Tensor of shape [1, H, W],
                inference from semantic prediction.
        Returns:
            ins_seg: A Tensor of shape [1, H, W] with value 0 represent stuff (not instance)
                and other positive values represent different instances.
            center_points: A Tensor of shape [1, K, 2] where K is the number of center points.
                The order of second dim is (y, x).
        """
        if center_points.size(0) == 0:
            return torch.zeros_like(sem_seg)
        # Generates a coordinate map, where each location is the coordinate of
        # that location.
        y_coord, x_coord = torch.meshgrid(
            torch.arange(self.height, dtype=offset.dtype, device=offset.device),
            torch.arange(self.width, dtype=offset.dtype, device=offset.device),
        )
        coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

        center_loc = coord + offset
        center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
        center_points = center_points.unsqueeze(1)  # [K, 1, 2]
        # Distance: [K, H*W].
        distance = torch.norm(center_points - center_loc, dim=-1)

        # Finds center with minimum distance at each location, offset by 1, to
        # reserve id=0 for stuff.
        ins_seg = torch.argmin(distance, dim=0).reshape((1, self.height, self.width)) + 1

        return thing_seg * ins_seg

    def merge_semantic_and_instance(self, sem_seg, ins_seg, thing_seg):
        """
        Post-processing for panoptic segmentation, by merging semantic segmentation
            label and class agnostic instance segmentation label.
        Args:
            sem_seg: A Tensor of shape [1, H, W], predicted category id for each pixel.
            ins_seg: A Tensor of shape [1, H, W], predicted instance id for each pixel.
            thing_seg: A Tensor of shape [1, H, W], predicted foreground mask.
        Returns:
            A Tensor of shape [1, H, W].
        """
        # In case thing mask does not align with semantic prediction.
        pan_seg = torch.zeros_like(sem_seg) + self.void_label
        is_thing = (ins_seg > 0) & (thing_seg > 0)

        # Paste thing by majority voting.
        instance_ids = torch.unique(ins_seg)
        for ins_id in instance_ids:
            if ins_id == 0:
                continue
            # Make sure only do majority voting within `thing_seg`.
            thing_mask = (ins_seg == ins_id) & is_thing
            if torch.nonzero(thing_mask).size(0) == 0:
                continue
            class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
            new_ins_id = class_id * self.label_divisor + ins_id
            pan_seg[thing_mask] = new_ins_id

        # Paste stuff to unoccupied area.
        class_ids = torch.unique(sem_seg)
        for class_id in class_ids:
            if class_id.item() in self.thing_ids:
                # thing class
                continue
            # Calculate stuff area.
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
            if stuff_mask.sum().item() >= self.stuff_area:
                pan_seg[stuff_mask] = class_id * self.label_divisor

        return pan_seg

    def get_instances(self, panoptic, center, sem_seg):
        instances = []
        semantic_prob = F.softmax(sem_seg, dim=0)
        for panoptic_label in torch.unique(panoptic):
            # if label == void label (-1) 
            if panoptic_label == -1:
                continue
            pred_classes = panoptic_label // self.label_divisor
            if pred_classes not in self.thing_ids:
                continue
            instance_ids = panoptic_label % self.label_divisor
            # get all instances of this class == panoptic label
            instances_of_class = Instances((self.height, self.width))
            # Evaluation code takes continuous id starting from 0
            instances_of_class.pred_classes = torch.tensor(
                [pred_classes], device=panoptic.device
            )
            # Used to match current and previous instances
            instances_of_class.instance_ids = torch.tensor(
                [instance_ids], device=panoptic.device
            )
            mask = panoptic == panoptic_label
            instances_of_class.pred_masks = mask.unsqueeze(0)
            # Average semantic probability
            sem_scores = semantic_prob[pred_classes, ...]
            sem_scores = torch.mean(sem_scores[mask])
            # Center point probability
            mask_indices = torch.nonzero(mask).float()
            center_y, center_x = (
                torch.mean(mask_indices[:, 0]),
                torch.mean(mask_indices[:, 1]),
            )
            center_scores = center[0, int(center_y.item()), int(center_x.item())]
            # Confidence score is semantic prob * center prob.
            instances_of_class.scores = torch.tensor(
                [sem_scores * center_scores], device=panoptic.device
            )
            # Get bounding boxes
            instances_of_class.pred_boxes = BitMasks(instances_of_class.pred_masks).get_bounding_boxes()
            instances.append(instances_of_class)
        # Concatenate the lists of instances of one class to one list cotaining all instances
        if len(instances) > 0:
            instances = Instances.cat(instances)
        return instances