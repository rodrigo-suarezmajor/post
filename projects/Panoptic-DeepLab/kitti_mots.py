import os
import cv2
from detectron2.data import DatasetCatalog

def get_kitti_mots(path):
    images = []
    for sequence in sorted(os.listdir(path)):
        image_list = sorted(os.listdir(os.path.join(path,sequence)))
        #get the height and width of the sequence
        height, width, _ = cv2.imread(
            os.path.join(path,sequence,image_list[0])
            ).shape
        for frame in image_list:
            image = {
                'file_name': os.path.join(path,sequence,frame), 
                'height': height, 
                'width': width, 
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

#get_kitti_mots("./datasets/kitti_mots/train/")
