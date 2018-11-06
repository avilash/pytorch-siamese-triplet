import os
import os.path

import cv2

import torch.utils.data
import torchvision.transforms as transforms

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, img3 = self.triplets[index]
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        img3 = cv2.imread(img3)
        img1 = cv2.resize(img1, (228, 228))
        img2 = cv2.resize(img2, (228, 228))
        img3 = cv2.resize(img3, (228, 228))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
