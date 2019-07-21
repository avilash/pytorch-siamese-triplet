import os
import random
from base_config import cfg

import cv2

class GOR(object):

    def __init__(self, classes):
    	self.__classes = classes

    def load(self):
    	self.__imgs_dict = {}
    	base_path = cfg.DATASETS.GOR.HOME
    	for obj_class in self.__classes:
            result_path = os.path.join(base_path, obj_class, "result")
            class_dict = []
            for pose in os.listdir(result_path):
            	pose_path = os.path.join(result_path, pose)
            	pose_crop_img_path = os.path.join(pose_path, "img_crop")
            	imgs = os.listdir(pose_crop_img_path)
            	for i,img in enumerate(imgs):
            		imgs[i] = os.path.join(pose_crop_img_path, img)
            	class_dict.append(imgs)
            self.__imgs_dict[str(obj_class)] = class_dict

    def getTriplet(self):
    	num_classes = len(self.__classes)
    	anchor_class_idx = random.randint(0, num_classes-1)
    	anchor_class = self.__classes[anchor_class_idx]
    	neg_class_idx = random.randint(0, num_classes-1)
    	while anchor_class == neg_class_idx:
    		neg_class_idx = random.randint(0, num_classes-1)
    	neg_class = self.__classes[neg_class_idx]

    	# Get anchor image and positive image
    	num_poses = len(self.__imgs_dict[anchor_class])
    	anchor_class_pose_idx = random.randint(0, num_poses-1)
    	num_imgs = len(self.__imgs_dict[anchor_class][anchor_class_pose_idx])
    	anchor_class_img_idx = random.randint(0, num_imgs-1)
    	anchor_class_img = self.__imgs_dict[anchor_class][anchor_class_pose_idx][anchor_class_img_idx]
    	pos_class_img_idx = random.randint(0, num_imgs-1)
    	while anchor_class_img_idx == pos_class_img_idx:
    		pos_class_img_idx = random.randint(0, num_imgs-1)
    	pos_class_img = self.__imgs_dict[anchor_class][anchor_class_pose_idx][pos_class_img_idx]

    	# Get negative image
    	num_poses = len(self.__imgs_dict[neg_class])
    	neg_class_pose_idx = random.randint(0, num_poses-1)
    	num_imgs = len(self.__imgs_dict[neg_class][neg_class_pose_idx])
    	neg_class_img_idx = random.randint(0, num_imgs-1)
    	neg_class_img = self.__imgs_dict[neg_class][neg_class_pose_idx][neg_class_img_idx]

    	return anchor_class_img, pos_class_img, neg_class_img