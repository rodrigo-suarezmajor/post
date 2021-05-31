import numpy as np
import torch
import pycocotools.mask as mask_util

class _Instance:
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

        instances = self.to_instance_dict(raw_instances)
        object_ids = np.zeros([len(raw_instances), 1], dtype=int)
        # When there's no previous instances (first image or no detection):
        if raw_prev_instances is None or len(raw_prev_instances) == 0:
            for i, instance_id in enumerate(instances):
                # Assign increasing id number
                self._num_objects += 1
                instances[instance_id].object_id = self._num_objects
                object_ids[i] = self._num_objects
            self._old_instances = instances.values() # TODO: ke  ep old instances
            raw_instances.object_ids = torch.tensor(object_ids, device=torch.device("cuda"))
            return raw_instances

        prev_instances = self.to_instance_dict(raw_prev_instances)
        # Compute iou:
        is_crowd = np.zeros((len(prev_instances),), dtype=np.bool)
        rles_old = [x.mask_rle for x in self._old_instances]
        rles_new = [x.mask_rle for x in prev_instances.values()]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.5

        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(prev_instances)), dtype="float32")

        # Only allow matching instances of the same class:
        for old_idx, old_inst in enumerate(self._old_instances):
            for idx, instance_id in enumerate(prev_instances.keys()):
                if old_inst.class_id != prev_instances[instance_id].class_id:
                    ious[old_idx, idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        prev_instance_ids = list(prev_instances.keys())
        for old_idx, old_inst in enumerate(self._old_instances):
            if max_iou_per_old[old_idx] > threshold:
                idx = matched_new_per_old[old_idx]
                instance_id = prev_instance_ids[idx]
                if prev_instances[instance_id].object_id is None:
                    prev_instances[instance_id].object_id = old_inst.object_id
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            old_inst.ttl -= 1
            if old_inst.ttl > 0:
                extra_instances.append(old_inst)

        # Assign object id to instances
        for i, instance_id in enumerate(instances.keys()):
            # Assign id of matched objects
            if instance_id in prev_instance_ids:
                object_id = prev_instances[instance_id].object_id
                if object_id:
                    instances[instance_id].object_id = prev_instances[instance_id].object_id
                    object_ids[i] = prev_instances[instance_id].object_id
                    continue
            # Assign new id to newly-detected instances:
            # Assign increasing id number
            self._num_objects += 1
            instances[instance_id].object_id = self._num_objects
            object_ids[i] = self._num_objects
        self._old_instances = list(instances.values()) + extra_instances
        raw_instances.object_ids = torch.tensor(object_ids, device=torch.device("cuda"))
        return raw_instances

    def to_instance_dict(self, raw_instances):
        raw_instances_cpu = raw_instances.to(torch.device("cpu"))
        instances = {}
        for i in range(len(raw_instances_cpu)):
            instance_id = int(raw_instances_cpu.instance_ids[i])
            class_id = int(raw_instances_cpu.pred_classes[i])
            # get mask_rle
            mask = raw_instances_cpu.pred_masks[i]
            mask_rle = mask_util.encode(np.asarray(mask[:, :, None], dtype=np.uint8, order="F"))[0]
            # save instance to instances dict
            instances[instance_id] = _Instance(class_id, mask_rle)
        return instances
    
    def reset(self):
        self._num_objects = 0
        self._old_instances = []