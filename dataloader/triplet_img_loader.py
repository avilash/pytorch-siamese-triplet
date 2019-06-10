import os
import os.path

import cv2
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        img1_details, img2_details, img3_details = self.triplets[index]
        img1 = cv2.imread(img1_details[0])
        img2 = cv2.imread(img2_details[0])
        img3 = cv2.imread(img3_details[0])

        try:
            bbox = img1_details[1]
            if bbox is not None:
                img1 = img1[bbox['top']:bbox['top']+bbox['height'], bbox['left']:bbox['left']+bbox['width']]
            img1 = cv2.resize(img1, (228, 228))
        except Exception, e:
            img1 = np.zeros((228, 228, 3), dtype=np.uint8)

        try:
            bbox = img2_details[1]
            if bbox is not None:
                img2 = img2[bbox['top']:bbox['top']+bbox['height'], bbox['left']:bbox['left']+bbox['width']]
            img2 = cv2.resize(img2, (228, 228))
        except Exception, e:
            img2 = np.zeros((228, 228, 3), dtype=np.uint8)
            
        try:
            bbox = img3_details[1]
            if bbox is not None:
                img3 = img3[bbox['top']:bbox['top']+bbox['height'], bbox['left']:bbox['left']+bbox['width']]
            img3 = cv2.resize(img3, (228, 228))
        except Exception, e:
            img3 = np.zeros((228, 228, 3), dtype=np.uint8)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        
        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
