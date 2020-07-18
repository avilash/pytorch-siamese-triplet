import _init_paths
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model.net import get_model
from dataloader.triplet_img_loader import get_loader
from utils.gen_utils import make_dir_if_not_exist
from utils.vis_utils import vis_with_paths, vis_with_paths_and_bboxes

from config.base_config import cfg, cfg_from_file


def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    exp_dir = os.path.join(args.result_dir, args.exp_name)
    make_dir_if_not_exist(exp_dir)

    # Build Model
    model = get_model(args, device)
    if model is None:
        return

    # Criterion and Optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value]}]
    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(params, lr=args.lr)

    # Train Test Loop
    for epoch in range(1, args.epochs + 1):
        # Init data loaders
        train_data_loader, test_data_loader = get_loader(args)
        # Test train
        test(test_data_loader, model, criterion)
        train(train_data_loader, model, criterion, optimizer, epoch)
        # Save model
        model_to_save = {
            "epoch": epoch + 1,
            'state_dict': model.state_dict(),
        }
        if epoch % args.ckp_freq == 0:
            file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
            save_checkpoint(model_to_save, file_name)


def train(data, model, criterion, optimizer, epoch):
    print("******** Training ********")
    total_loss = 0
    model.train()
    for batch_idx, img_triplet in enumerate(data):
        anchor_img, pos_img, neg_img = img_triplet
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_step = args.train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
            total_loss = 0
    print("****************")


def test(data, model, criterion):
    print("******** Testing ********")
    with torch.no_grad():
        model.eval()
        accuracies = [0, 0, 0]
        acc_threshes = [0, 0.2, 0.5]
        total_loss = 0
        for batch_idx, img_triplet in enumerate(data):
            anchor_img, pos_img, neg_img = img_triplet
            anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
            anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
            E1, E2, E3 = model(anchor_img, pos_img, neg_img)
            dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
            dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

            target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
            target = target.to(device)
            target = Variable(target)

            loss = criterion(dist_E1_E2, dist_E1_E3, target)
            total_loss += loss

            for i in range(len(accuracies)):
                prediction = (dist_E1_E3 - dist_E1_E2 - args.margin * acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc
        print('Test Loss: {}'.format(total_loss / len(data)))
        for i in range(len(accuracies)):
            print(
                'Test Accuracy with diff = {}% of margin: {}'.format(acc_threshes[i] * 100, accuracies[i] / len(data)))
    print("****************")


def save_checkpoint(state, file_name):
    torch.save(state, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='exp0', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                        help="List of GPU Devices to train on")
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--ckp_freq', type=int, default=1, metavar='N',
                        help='Checkpoint Frequency (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                        help='margin for triplet loss (default: 1.0)')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
                        help='Dataset (default: mnist)')

    parser.add_argument('--num_train_samples', type=int, default=50000, metavar='M',
                        help='number of training samples (default: 3000)')
    parser.add_argument('--num_test_samples', type=int, default=10000, metavar='M',
                        help='number of test samples (default: 1000)')

    parser.add_argument('--train_log_step', type=int, default=100, metavar='M',
                        help='Number of iterations after which to log the loss')

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
