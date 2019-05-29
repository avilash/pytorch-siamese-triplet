import os
import os.path

import numpy as np

import torch.utils.data
import torchvision.transforms as transforms

class TripletMNISTLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        img1, img2, img3 = self.triplets[index] 
        img1 = np.expand_dims(img1, axis=2) 
        img2 = np.expand_dims(img2, axis=2) 
        img3 = np.expand_dims(img3, axis=2) 

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
