import os
import numpy as np
import torch
import pycocotools.mask as mask_util
from detectron2.data import MetadataCatalog


class _Instance:
    """
    Used to store data about detected objects in video frame,
    in order to enable simple iou tracking.

    Attributes:
        class_id (int):
        bbox (tuple[float]):
        mask_rle (dict):
        object_id (int):
    """

    __slots__ = ["class_id", "mask_rle", 'object_id', 'ttl']

    def __init__(self, class_id, mask_rle, object_id=None,  ttl=8):
        self.class_id = class_id
        self.mask_rle = mask_rle
        self.object_id = object_id
        self.ttl = ttl


class KittiMotsEvaluator():
    def __init__(self, dataset_name):
        self._num_objects = 0
        self._old_instances = []
        self._output_dir = './output/data'
        self._cpu_device = torch.device("cpu")
        self._thing_classes = MetadataCatalog.get(dataset_name).thing_classes

    def reset(self):
        for f in os.listdir(self._output_dir):
            os.remove(os.path.join(self._output_dir, f))

    
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            #check if there are instances in this frame
            if "instances" not in output:
                return
            
            # create internal instance object (used for iou tracking) for each instance
            raw_instances = output["instances"].to(self._cpu_device)
    
            instances = []
            for i in range(len(raw_instances)):
                pred_class = raw_instances.pred_classes[i]
                object_id = raw_instances.object_ids[i][0] if raw_instances.object_ids is not None else None
                kitti_mots_class = self._to_kitti_mots(pred_class)
                # check if the class is tracked in kitti mots
                if kitti_mots_class is None: 
                    continue
                # get mask_rle
                mask = raw_instances.pred_masks[i]
                mask_rle = mask_util.encode(np.asarray(mask[:, :, None], dtype=np.uint8, order="F"))[0]
                # save instance to instances
                instances.append(_Instance(kitti_mots_class, mask_rle, object_id))
            # check if there are kitti mots instances in the frame
            if instances == []:
                return

            if instances[0].object_id is None:
                # perform iou tracking (assigns object id)
                self._iou_tracking(instances)

            # get the time frame for the output
            frame = os.path.splitext(os.path.basename(input['file_name']))[0]
            time_frame = frame.lstrip('0')
            if time_frame == '':
                time_frame = '0'
            # get the height and width
            height = input['height']
            width = input['width']
            
            
            with open(os.path.join(self._output_dir,input['sequence']) + '.txt', 'a') as fout:
                for i in range(len(instances)):
                    object_id = instances[i].object_id
                    class_id = instances[i].class_id
                    # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                    # json writer which always produces strings cannot serialize a bytestream
                    # unless you decode it. Thankfully, utf-8 works out (which is also what
                    # the pycocotools/_mask.pyx does).
                    mask_rle = instances[i].mask_rle["counts"].decode("utf-8")
                    fout.write(
                        "{} {} {} {} {} {}\n"
                        .format(time_frame, object_id, class_id, height,  width,  mask_rle)
                    )
            fout.close()
            

    def evaluate(self):
        pass

    def _to_kitti_mots(self, pred_class):
        thing_class = self._thing_classes[pred_class]
        if thing_class == "car":
            return 1
        elif thing_class in "person":
            return 2
        else:
            return None

    def _iou_tracking(self, instances):
        """
        Naive tracking heuristics to assign same object id to the same instance,
        will update the internal state of tracked instances.
        """

        # Compute iou:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        assert instances[0].mask_rle is not None
        rles_old = [x.mask_rle for x in self._old_instances]
        rles_new = [x.mask_rle for x in instances]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.5

        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same class:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.class_id != new.class_id:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
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

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.object_id is None:
                # Assign increasing id number
                self._num_objects += 1
                inst.object_id = self._num_objects
        self._old_instances = instances[:] + extra_instances