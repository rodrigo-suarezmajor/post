import os
import cv2
from detectron2.data import DatasetCatalog

def get_kitti_mots(path):
    images = []
    instances_path = '../instances_txt/'
    for sequence in sorted(os.listdir(path)):
        image_list = sorted(os.listdir(os.path.join(path,sequence)))
        # Get the height and width of the sequence
        height, width, _ = cv2.imread(
            os.path.join(path,sequence,image_list[0])
            ).shape

        # Get instances of the whole sequence
        f = open(os.path.join(path, instances_path, sequence + ".txt"), "r")
        instances = {}
        for line in f:
            line = line.strip()
            fields = line.split()
            frame = int(fields[0])
            object_id = int(fields[1]) % 1000 
            class_id = int(fields[2])
            mask = {
                'size': [int(fields[3]), int(fields[4])], 
                'counts': fields[5].encode(encoding='UTF-8')
                }
            instance = {
                'object_id:': object_id,
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
            
            # Check if there are annotations/instances in this frame
            frame = image_name.lstrip('0').rstrip('.png')
            if frame == '':
                frame = 0
            frame = int(frame)
            if frame in instances:
                annotations = instances[frame]
            else:
                annotations = []

            image = {
                'file_name': os.path.join(path, sequence, image_name), 
                'height': height, 
                'width': width, 
                'annotations': annotations, 
                'sequence': sequence
                }
            images.append(image)
    return images

def register():
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "kitti_mots_" + d,
            lambda d=d: get_kitti_mots("./datasets/kitti_mots/" + d)
            )

# get_kitti_mots("./datasets/kitti_mots/train/")
