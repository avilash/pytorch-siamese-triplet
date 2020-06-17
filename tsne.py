import _init_paths
import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model import net, embedding

from utils.gen_utils import make_dir_if_not_exist

from config.base_config import cfg, cfg_from_file

import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    exp_dir = os.path.join("data", args.exp_name)
    make_dir_if_not_exist(exp_dir)

    if args.pkl is not None:
        input_file = open(args.pkl, 'rb')
        final_data = pickle.load(input_file)
        input_file.close()
        embeddings = final_data['embeddings']
        labels = final_data['labels']
        vis_tSNE(embeddings, labels)
    else:
        embeddingNet = None
        if (args.dataset == 's2s') or (args.dataset == 'vggface2'):
            embeddingNet = embedding.EmbeddingResnet()
        elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
            embeddingNet = embedding.EmbeddingLeNet()
        else:
            print("Dataset {} not supported ".format(args.dataset))
            return

        model_dict = None
        if args.ckp is not None:
            if os.path.isfile(args.ckp):
                print("=> Loading checkpoint '{}'".format(args.ckp))
                try:
                    model_dict = torch.load(args.ckp)['state_dict']
                except Exception:
                    model_dict = torch.load(args.ckp, map_location='cpu')['state_dict']
                print("=> Loaded checkpoint '{}'".format(args.ckp))
            else:
                print("=> No checkpoint found at '{}'".format(args.ckp))
                return
        else:
            print("Please specify a model")
            return

        model_dict_mod = {}
        for key, value in model_dict.items():
            new_key = '.'.join(key.split('.')[2:])
            model_dict_mod[new_key] = value
        model = embeddingNet.to(device)
        model.load_state_dict(model_dict_mod)

        data_loader = None
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        if (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
            transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = None
            if args.dataset == 'mnist':
                train_dataset = MNIST('data/MNIST', train=True, download=True, transform=transform)
            if args.dataset == 'fmnist':
                train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform)
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        else:
            print("Dataset {} not supported ".format(args.dataset))
            return

        embeddings, labels = generate_embeddings(data_loader, model)

        final_data = {
            'embeddings': embeddings,
            'labels': labels
        }

        dst_dir = os.path.join('data', args.exp_name, 'tSNE')
        make_dir_if_not_exist(dst_dir)

        output_file = open(os.path.join(dst_dir, 'tSNE.pkl'), 'wb')
        pickle.dump(final_data, output_file)
        output_file.close()

        vis_tSNE(embeddings, labels)


def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_imgs = Variable(batch_imgs.to(device))
            bacth_E = model(batch_imgs)
            bacth_E = bacth_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels


def vis_tSNE(embeddings, labels):
    num_samples = args.tSNE_ns if args.tSNE_ns < embeddings.shape[0] else embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(embeddings[0:num_samples, :])

    fig, ax = plt.subplots()

    x, y = X_embedded[:, 0], X_embedded[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    sc = ax.scatter(x, y, c=labels[0:num_samples], cmap=mpl.colors.ListedColormap(colors))
    plt.colorbar(sc)
    plt.savefig(os.path.join('data', args.exp_name, 'tSNE', 'tSNE_' + str(num_samples) + '.jpg'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
                        help='Dataset (default: mnist)')

    parser.add_argument('--pkl', default=None, type=str,
                        help='Path to load embeddings')

    parser.add_argument('--tSNE_ns', default=5000, type=int,
                        help='Num samples to create a tSNE visualisation')

    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()
