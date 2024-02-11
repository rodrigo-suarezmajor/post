import os
from pycocotools import mask as pycoco_mask
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_rob_mots(path, train):
    images = []
    instances_path = 'instances_txt/'
    for dataset in sorted(os.listdir(path)):
        seq_path = os.path.join(path, dataset)
        for sequence in sorted(os.listdir(seq_path)):
            image_list = sorted(os.listdir(os.path.join(seq_path,sequence)))
            # Get the height and width of the sequence
            height, width, _ = cv2.imread(
                os.path.join(seq_path,sequence,image_list[0])
                ).shape

            instances = {}
            if train:
                # Get instances of the whole sequence
                f = open(os.path.join(seq_path, '..',  instances_path, sequence + ".txt"), "r")
                for line in f:
                    line = line.strip()
                    fields = line.split()
                    frame = int(fields[0])
                    object_id = int(fields[1]) % 1000 
                    class_id = int(fields[2])
                    # Skip don't care region (id=10) for now
                    if class_id == 10:
                        continue
                    # map person to id in coco dataset
                    if class_id == 1:
                        class_id = 0
                    mask = {
                        'size': [int(fields[3]), int(fields[4])], 
                        'counts': fields[5]
                        }
                    bbox = pycoco_mask.toBbox(mask)
                    instance = {
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'object_id': object_id,
                        'category_id': class_id,
                        'segmentation': mask
                        }    
                    if frame not in instances:
                        instances[frame] = [instance]
                    else:
                        instances[frame].append(instance)
                f.close()

            # Generate output dict    
            for image_name in image_list:
                file_name = os.path.join(seq_path, sequence, image_name)
                # Check if there are annotations/instances in this frame
                frame = os.path.splitext(image_name)[0].lstrip('0')
                if frame == '':
                    frame = 0
                    prev_file_name = None
                frame = int(frame)
                if frame in instances:
                    annotations = instances[frame]
                else:
                    annotations = []

                # Get the previous annotations
                prev_frame = frame - 1
                if prev_frame >= 0 and prev_frame in instances:
                    prev_annotations = instances[prev_frame]
                else:
                    prev_annotations = []

                image = {
                    'file_name': file_name, 
                    'prev_file_name': prev_file_name, 
                    'height': height, 
                    'width': width, 
                    'annotations': annotations, 
                    'prev_annotations': prev_annotations,
                    'dataset': dataset,
                    'sequence': sequence
                    }
                images.append(image)
                prev_file_name = file_name
    return images

def register_rob_mots(train=False, val=False):
    datasets = []
    if train:
        datasets.append('train')
    if val:
        datasets.append('val')
    for d in datasets:
        DatasetCatalog.register(
            "rob_mots_" + d,
            lambda d=d: get_rob_mots("./datasets/rob_mots/train/", train=train)
            )
        coco_meta = MetadataCatalog.get("coco_2017_" + d + "_panoptic").as_dict()
        coco_meta.pop("name", None)
        MetadataCatalog.get("rob_mots_" + d).set(**coco_meta)

