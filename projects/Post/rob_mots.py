import os
from pycocotools import mask as pycoco_mask
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_rob_mots(base_path, split):
    images = []
    path = os.path.join(base_path, split)
    for dataset in sorted(os.listdir(path)):
        seq_path = os.path.join(path, dataset)
        for sequence in sorted(os.listdir(seq_path)):
            image_list = sorted(os.listdir(os.path.join(seq_path,sequence)))
            # Get the height and width of the sequence
            height, width, _ = cv2.imread(
                os.path.join(seq_path,sequence,image_list[0])
                ).shape

            annotations = {}
            if split == 'train':
                # Get annotations of the whole sequence
                f = open(os.path.join(base_path, "gt", dataset, "data", sequence + ".txt"), "r")
                for line in f:
                    line = line.strip()
                    fields = line.split()
                    frame = int(fields[0])
                    object_id = int(fields[1])  
                    class_id = int(fields[2])
                    # Skip ignore region (id >= 100) for now
                    if class_id >= 100:
                        continue
                    mask = {
                        'size': [int(fields[4]), int(fields[5])], 
                        'counts': fields[6]
                        }
                    bbox = pycoco_mask.toBbox(mask)
                    annotation = {
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'object_id': object_id,
                        'category_id': class_id,
                        'segmentation': mask
                        }    
                    if frame not in annotations:
                        annotations[frame] = [annotation]
                    else:
                        annotations[frame].append(annotation)
                f.close()

            # Generate output dict    
            for image_name in image_list:
                file_name = os.path.join(seq_path, sequence, image_name)
                # Check if there are annotations/annotations in this frame
                frame = os.path.splitext(image_name)[0].lstrip('0')
                if frame == '':
                    frame = 0
                    prev_file_name = None
                frame = int(frame)

                # Get annotations
                if frame in annotations:
                    annotations = annotations[frame]
                else:
                    annotations = []
                # Get the previous annotations
                prev_frame = frame - 1
                if prev_frame >= 0 and prev_frame in annotations:
                    prev_annotations = annotations[prev_frame]
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
                    'sequence': sequence,
                    }
                images.append(image)
                prev_file_name = file_name
    return images

def register_rob_mots():
    for d in ['train', 'val', 'test']:
        DatasetCatalog.register(
            "rob_mots_" + d,
            lambda d=d: get_rob_mots("./datasets/rob_mots/", d)
            )
        coco_meta = MetadataCatalog.get("coco_2017_" + d + "_panoptic").as_dict()
        coco_meta.pop("name", None)
        MetadataCatalog.get("rob_mots_" + d).set(**coco_meta)
