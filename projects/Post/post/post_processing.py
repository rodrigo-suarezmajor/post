# Copyright (c) Facebook, Inc. and its affiliates.
# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py  # noqa

from collections import Counter
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import BitMasks, Instances
import torch
import numpy as np
import torch.nn.functional as F
import pycocotools.mask as mask_util

class _PrevImage:
    """
    Used to store data about the segmentation in the previous image
    to generate panoptic image of the previous fram with current centers
    """
    __slots__ = ["sem_seg", "thing_seg", "panoptic"]
    def __init__(self, sem_seg=None, thing_seg=None, panoptic=None):
        self.sem_seg = sem_seg
        self.thing_seg = thing_seg
        self.panoptic = panoptic

class Instance:
    """
    Used to store data about detected objects in Previous Frame
    to enable iou tracking
    """

    __slots__ = ["class_id", "bbox", "mask_rle", 'object_id', 'ttl']

    def __init__(self, class_id, mask_rle, object_id=None, ttl=8):
        self.class_id = class_id
        self.mask_rle = mask_rle
        self.object_id = object_id
        self.ttl = ttl


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
                semantic id * label_divisor + instance_id.
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
        self.prev_instances = []
        self.num_objects = 0

    def __call__(self, sem_seg, center, offset, prev_offset, input, output_size):
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
        (height, width) = (input.get("height"), input.get("width"))
        sem_seg = sem_seg_postprocess(sem_seg, output_size, height, width)
        center = sem_seg_postprocess(center, output_size, height, width)
        offset = sem_seg_postprocess(offset, output_size, height, width)
        prev_offset = sem_seg_postprocess(prev_offset, output_size, height, width)

        # For semantic segmentation evaluation.
        processed_result['sem_seg'] = sem_seg
        # Post-processing to get panoptic segmentation.
        panoptic_image, _ = self.get_panoptic_segmentation(
            sem_seg.argmax(dim=0, keepdim=True),
            center,
            offset,
            prev_offset
        )
        # For panoptic segmentation evaluation.
        processed_result['panoptic_seg'] = (panoptic_image, None)
  
        if not self.predict_instances:
            return processed_result
        
        # For instance segmentation evaluation.
        prev_instances = self.get_instances(
            self.prev_image.panoptic.squeeze(0),
            center,
            self.prev_image.sem_seg,
            height,
            width
            )
        self.iou_tracking(prev_instances)   
        instances = self.get_instances(panoptic_image, center, sem_seg, height, width)
        if instances:
            processed_result["instances"] = instances


    def get_panoptic_segmentation(self, sem_seg, center, offset, prev_offset, foreground_mask=None):
        if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
            raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))
        if center.dim() != 3:
            raise ValueError(
                "Center prediction with un-supported dimension: {}.".format(center.dim())
            )
        if offset.dim() != 3:
            raise ValueError("Offset prediction with un-supported dimension: {}.".format(offset.dim()))
        if foreground_mask is not None:
            if self.foreground_mask.dim() != 3 and foreground_mask.size(0) != 1:
                raise ValueError(
                    "Foreground prediction with un-supported shape: {}.".format(sem_seg.size())
                )
            thing_seg = foreground_mask
        else:
            # inference from semantic segmentation
            thing_seg = torch.zeros_like(sem_seg)
            for thing_class in list(self.thing_ids):
                thing_seg[sem_seg == thing_class] = 1

        instance, center_points = self.get_instance_segmentation(sem_seg, center, offset, thing_seg)
        
        # Generate panoptic of previous image with current center points
        if self.prev_image.sem_seg is not None:
            prev_instance, _ = self.get_instance_segmentation(sem_seg, center, prev_offset, self.prev_image.thing_seg)
            prev_panoptic = self.merge_semantic_and_instance(self.prev_image.sem_seg, prev_instance, self.prev_image.thing_seg)


        panoptic = self.merge_semantic_and_instance(sem_seg, instance, thing_seg)

        # Update previous instance
        self.prev_image.sem_seg = sem_seg
        self.prev_image.thing_seg = thing_seg
        self.prev_image.panoptic = panoptic
        return panoptic.squeeze(0), center_points

    def get_instances(self, panoptic_image, center, sem_seg, height, width):
        instances = []
        semantic_prob = F.softmax(sem_seg, dim=0)
        panoptic_image = panoptic_image.cpu().numpy()
        for panoptic_label in np.unique(panoptic_image):
            if panoptic_label == -1:
                continue
            pred_class = panoptic_label // self.label_divisor
            if pred_class not in self.thing_ids:
                continue

            instances_per_class = Instances((height, width))
            # Evaluation code takes continuous id starting from 0
            instances_per_class.pred_classes = torch.tensor(
                [pred_class], device=panoptic_image.device
            )
            mask = panoptic_image == panoptic_label
            instances_per_class.pred_masks = mask.unsqueeze(0)
            # Average semantic probability
            sem_scores = semantic_prob[pred_class, ...]
            sem_scores = torch.mean(sem_scores[mask])
            # Center point probability
            mask_indices = torch.nonzero(mask).float()
            center_y, center_x = (
                torch.mean(mask_indices[:, 0]),
                torch.mean(mask_indices[:, 1]),
            )
            # Used to match current and previous instances 
            instances_per_class.center = [int(center_y.item()), int(center_x.item())]
            center_scores = center[0, int(center_y.item()), int(center_x.item())]
            # Confidence score is semantic prob * center prob.
            instances_per_class.scores = torch.tensor(
                [sem_scores * center_scores], device=panoptic_image.device
            )
            # Get bounding boxes
            instances_per_class.pred_boxes = BitMasks(instances_per_class.pred_masks).get_bounding_boxes()
            instances.append(instances_per_class)
        # Concatenate the lists of instances per class to one list
        if len(instances) > 0:
            instances = Instances.cat(instances)
        return instances

    def get_instance_segmentation(self, sem_seg, center, offset, thing_seg):
        """
        Post-processing for instance segmentation, gets class agnostic instance id.
        Args:
            sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
            center: A Tensor of shape [1, H, W] of raw center heatmap output.
            offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
                second dim is (offset_y, offset_x).
            thing_seg: A Tensor of shape [1, H, W], predicted foreground mask,
                if not provided, inference from semantic prediction.
        Returns:
            A Tensor of shape [1, H, W] with value 0 represent stuff (not instance)
                and other positive values represent different instances.
            A Tensor of shape [1, K, 2] where K is the number of center points.
                The order of second dim is (y, x).
        """
        center_points = self.find_instance_center(center)
        if center_points.size(0) == 0:
            return torch.zeros_like(sem_seg), center_points.unsqueeze(0)
        ins_seg = self.group_pixels(center_points, offset)
        return thing_seg * ins_seg, center_points.unsqueeze(0)

    def find_instance_center(self, center):
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
        center_heatmap_max_pooled = F.max_pool2d(
            center, kernel_size=self.nms_kernel, stride=1, padding=nms_padding
        )
        center[center != center_heatmap_max_pooled] = -1

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


    def group_pixels(self, center_points, offset):
        """
        Gives each pixel in the image an instance id.
        Args:
            center_points: A Tensor of shape [K, 2] where K is the number of center points.
                The order of second dim is (y, x).
            offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
                second dim is (offset_y, offset_x).
        Returns:
            A Tensor of shape [1, H, W] with values in range [1, K], which represents
                the center this pixel belongs to.
        """
        height, width = offset.size()[1:]

        # Generates a coordinate map, where each location is the coordinate of
        # that location.
        y_coord, x_coord = torch.meshgrid(
            torch.arange(height, dtype=offset.dtype, device=offset.device),
            torch.arange(width, dtype=offset.dtype, device=offset.device),
        )
        coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

        center_loc = coord + offset
        center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
        center_points = center_points.unsqueeze(1)  # [K, 1, 2]

        # Distance: [K, H*W].
        distance = torch.norm(center_points - center_loc, dim=-1)

        # Finds center with minimum distance at each location, offset by 1, to
        # reserve id=0 for stuff.
        instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
        return instance_id



    def merge_semantic_and_instance(self, sem_seg, ins_seg, semantic_thing_seg):
        """
        Post-processing for panoptic segmentation, by merging semantic segmentation
            label and class agnostic instance segmentation label.
        Args:
            sem_seg: A Tensor of shape [1, H, W], predicted category id for each pixel.
            ins_seg: A Tensor of shape [1, H, W], predicted instance id for each pixel.
            semantic_thing_seg: A Tensor of shape [1, H, W], predicted foreground mask.
        Returns:
            A Tensor of shape [1, H, W].
        """
        # In case thing mask does not align with semantic prediction.
        pan_seg = torch.zeros_like(sem_seg) + self.void_label
        is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

        # Keep track of instance id for each class.
        class_id_tracker = Counter()

        # Paste thing by majority voting.
        instance_ids = torch.unique(ins_seg)
        for ins_id in instance_ids:
            if ins_id == 0:
                continue
            # Make sure only do majority voting within `semantic_thing_seg`.
            thing_mask = (ins_seg == ins_id) & is_thing
            if torch.nonzero(thing_mask).size(0) == 0:
                continue
            class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
            class_id_tracker[class_id.item()] += 1
            new_ins_id = class_id_tracker[class_id.item()]
            pan_seg[thing_mask] = class_id * self.label_divisor + new_ins_id

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

    def iou_tracking(self, raw_instances):
        """
        iou tracking between the instance masks of the previous frame,
        with current and previous center
        """
        instances = []
        for i in range(len(raw_instances)):
            # get mask_rle
            mask = raw_instances.pred_masks[i]
            mask_rle = mask_util.encode(np.asarray(mask[:, :, None], dtype=np.uint8, order="F"))[0]
            # save instance to instances
            instances.append(Instance(raw_instances.pred_classes[i], mask_rle))
        # Compute iou:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        assert instances[0].mask_rle is not None
        rles_old = [x.mask_rle for x in self.prev_instances]
        rles_new = [x.mask_rle for x in instances]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.5

        if len(ious) == 0:
            ious = np.zeros((len(self.prev_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same class:
        for old_idx, old in enumerate(self.prev_instances):
            for new_idx, new in enumerate(instances):
                if old.class_id != new.class_id:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self.prev_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].object_id is None:
                    instances[newidx].object_id = inst.object_id
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign new id to newly-detected instances:
        for inst in instances:
            if inst.object_id is None:
                # Assign increasing id number
                self.num_objects += 1
                inst.object_id = self.num_objects
        return extra_instances