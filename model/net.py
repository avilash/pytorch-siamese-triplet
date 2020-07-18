import os
import torch
import torch.nn as nn

from model import embedding


class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1, i2, i3):
        E1 = self.embeddingNet(i1)
        E2 = self.embeddingNet(i2)
        E3 = self.embeddingNet(i3)
        return E1, E2, E3


def get_model(args, device):
    # Model
    embeddingNet = None
    if (args.dataset == 'custom') or (args.dataset == 'vggface2'):
        embeddingNet = embedding.EmbeddingResnet()
    elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
        embeddingNet = embedding.EmbeddingLeNet()
    else:
        print("Dataset %s not supported " % args.dataset)
        return None

    model = TripletNet(embeddingNet)
    model = nn.DataParallel(model, device_ids=args.gpu_devices)
    model = model.to(device)

    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))

    return model
