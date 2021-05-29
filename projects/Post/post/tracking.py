import numpy as np
import torch
import pycocotools.mask as mask_util

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


class IouTracking:
    """
    iou tracking between the instance masks of the previous image, 
    and the 'new' previous instance masks with the current center
    """
    def __init__(self):
        self._num_objects = 0
        self._old_instances = []

    def __call__(self, raw_instances, raw_prev_instances):
        return raw_instances
        # todo
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
        return instances