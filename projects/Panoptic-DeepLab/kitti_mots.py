import os
from detectron2.data import DatasetCatalog

path = "./datasets/kitti_mots/training"
height = 375
width = 1242

def kitti_mots():
    images = []
    for sequence in os.listdir(path):
        for frame in os.listdir(os.path.join(path,sequence)):
            image_id = sequence + frame
            image = {'file_name': os.path.join(path,sequence,frame), 'height': height, 'width': width, 'image_id': image_id}
            images.append(image)
    return images

def register():
    DatasetCatalog.register("kitti_mots", kitti_mots)
