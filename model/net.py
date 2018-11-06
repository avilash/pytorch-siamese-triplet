import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        # dist_a = F.pairwise_distance(E1, E2, 2)
        # dist_b = F.pairwise_distance(E1, E3, 2)
        return E1, E2, E3