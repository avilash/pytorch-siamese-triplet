import _init_paths
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from model import net, embedding
from dataloader import gor, s2s, mnist
from triplet_img_loader import TripletImageLoader
from triplet_mnist_loader import TripletMNISTLoader

from gen_utils import make_dir_if_not_exist

from base_config import cfg, cfg_from_file

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualise(imgs, txts, dst):
	f, axs = plt.subplots(1, len(imgs), figsize=(24, 9))
	f.tight_layout()
	for ax, img, txt in zip(axs, imgs, txts):
		ax.imshow(img)
		ax.set_title(txt, fontsize=20)
	plt.subplots_adjust(left=0., right=1, top=0.95, bottom=0.)
	if dst is not "":
		plt.savefig(dst)
	plt.show()

def vis_with_paths(img_paths, txts, dst):
	imgs = []
	for img_path in img_paths:
		imgs.append(cv2.imread(img_path))
	visualise(imgs, txts, dst)

def vis_with_paths_and_bboxes(img_details, txts, dst):
	imgs = []
	for img_path, bbox in img_details:
		img = cv2.imread(img_path)
		if bbox is not None:
			img = img[bbox['top']:bbox['top']+bbox['height'], bbox['left']:bbox['left']+bbox['width']]
		imgs.append(img)
	visualise(imgs, txts, dst)

def main():
	
	torch.manual_seed(1)
	if args.cuda:
		torch.cuda.set_device(1)
		torch.cuda.manual_seed(1)

	exp_dir = os.path.join("data", args.exp_name)
	make_dir_if_not_exist(exp_dir)

	embeddingNet = None
	if args.dataset == 's2s':
		embeddingNet = embedding.EmbeddingResnet()
	elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
		embeddingNet = embedding.EmbeddingLeNet()
	else:
		print "Dataset %s not supported "%(args.dataset)
		return

	model = net.TripletNet(embeddingNet)
	if args.cuda:
		model = model.cuda()

	if args.ckp:
		if os.path.isfile(args.ckp):
			print("=> Loading checkpoint '{}'".format(args.ckp))
			checkpoint = torch.load(args.ckp)
			model.load_state_dict(checkpoint['state_dict'])
			print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp)) 

	cudnn.benchmark = True

	params = []
	for key, value in dict(model.named_parameters()).items():
		if value.requires_grad:
			params += [{'params':[value]}]

	criterion = torch.nn.MarginRankingLoss(margin = args.margin)
	optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

	for epoch in range(1, args.epochs + 1):
		train_data_loader, test_data_loader = sample_data()
		train(train_data_loader, model, criterion, optimizer, epoch)
		test(test_data_loader, model)
		model_to_save = {
			"epoch" : epoch + 1,
			'state_dict': model.state_dict(),
		}
		file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
		save_checkpoint(model_to_save, file_name)

def sample_data():
	train_data_loader = None
	test_data_loader = None
	
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	if args.dataset == 's2s':
		train_triplets = []
		test_triplets = []
		triplet_loader = s2s.S2S()
		train_len, test_len = triplet_loader.load()
		print "Train pairs = %d"%(train_len)
		print "Test pairs = %d"%(test_len)
		sku, product_img_details, pos_img_details, neg_img_details = triplet_loader.getTriplet()
		# vis_with_paths_and_bboxes([product_img_details, pos_img_details, neg_img_details], [sku, sku, sku], "")
		# return
		for i in range(args.num_train_samples):
			sku, product_img_details, pos_img_details, neg_img_details = triplet_loader.getTriplet()
			train_triplets.append([product_img_details, pos_img_details, neg_img_details])
		for i in range(args.num_test_samples):
			sku, product_img_details, pos_img_details, neg_img_details = triplet_loader.getTriplet(set="test")
			test_triplets.append([product_img_details, pos_img_details, neg_img_details])

		train_data_loader = torch.utils.data.DataLoader(
		TripletImageLoader(train_triplets,
		               transform=transforms.Compose([
		                   transforms.ToTensor(),
		                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		               ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
		test_data_loader = torch.utils.data.DataLoader(
		TripletImageLoader(test_triplets,
		               transform=transforms.Compose([
		                   transforms.ToTensor(),
		                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		               ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	
	elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
		train_triplets = []
		test_triplets = []
		if args.dataset == 'mnist':
			train_dataset = MNIST('data/MNIST', train=True, download=True)
			test_dataset = MNIST('data/MNIST', train=False, download=True)
		if args.dataset == 'fmnist':
			train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True)
			test_dataset = FashionMNIST('data/FashionMNIST', train=False, download=True)
		triplet_loader = mnist.MNIST_DS(train_dataset, test_dataset)
		triplet_loader.load()
		pos_label, neg_label, pos_anchor, pos_img, neg_img = triplet_loader.getTriplet()
		# visualise([pos_anchor, pos_img, neg_img], ["", str(pos_label), str(neg_label)], "")
		# return
		for i in range(args.num_train_samples):
			pos_label, neg_label, pos_anchor, pos_img, neg_img = triplet_loader.getTriplet()
			train_triplets.append([pos_anchor, pos_img, neg_img])
		for i in range(args.num_test_samples):
			pos_label, neg_label, pos_anchor, pos_img, neg_img = triplet_loader.getTriplet(set="test")
			test_triplets.append([pos_anchor, pos_img, neg_img])

		train_data_loader = torch.utils.data.DataLoader(
		TripletMNISTLoader(train_triplets,
		               transform=transforms.Compose([
		                   transforms.ToTensor(),
		                   transforms.Normalize((0.485,), (0.229,))
		               ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
		test_data_loader = torch.utils.data.DataLoader(
		TripletMNISTLoader(test_triplets,
		               transform=transforms.Compose([
		                   transforms.ToTensor(),
		                   transforms.Normalize((0.485,), (0.229,))
		               ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)

	return train_data_loader, test_data_loader

def train(data, model, criterion, optimizer, epoch):
	print ("Training ...")
	total_loss = 0
	model.train()
	for batch_idx, img_triplet in enumerate(data):
		anchor_img, pos_img, neg_img = img_triplet
		if args.cuda:
			anchor_img, pos_img, neg_img = anchor_img.cuda(), pos_img.cuda(), neg_img.cuda()
		anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
		E1, E2, E3 = model(anchor_img, pos_img, neg_img)
		dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
		dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

		target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
		if args.cuda:
			target = target.cuda()
		target = Variable(target)
		
		#Calculate loss
		loss_triplet = criterion(dist_E1_E2, dist_E1_E3, target)
		loss_embedd = E1.norm(2) + E2.norm(2) + E3.norm(2)
		loss = loss_triplet + 0.001*loss_embedd
		total_loss += loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		log_step = args.train_log_step
		if (batch_idx%log_step == 0) and (batch_idx!=0):
			print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss/log_step))
			total_loss = 0

def test(data, model):
	print ("Testing ...")
	model.eval()
	accuracy = 0
	for batch_idx, img_triplet in enumerate(data):
		anchor_img, pos_img, neg_img = img_triplet
		if args.cuda:
			anchor_img, pos_img, neg_img = anchor_img.cuda(), pos_img.cuda(), neg_img.cuda()
		anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
		E1, E2, E3 = model(anchor_img, pos_img, neg_img)
		dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
		dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

		prediction = (dist_E1_E3 - dist_E1_E2 - args.margin/2.0).cpu().data
		prediction = prediction.view(prediction.numel())
		prediction = (prediction > 0).float()
		batch_acc = prediction.sum()*1.0/prediction.numel()
		accuracy += batch_acc
	accuracy /= len(data)
	print('Test Accuracy: {}'.format(accuracy))

def save_checkpoint(state, file_name):
    torch.save(state, file_name)

if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
	parser.add_argument('--exp_name', default='exp0', type=str,
	                help='name of experiment')
	parser.add_argument('--cuda', action='store_true', default=False,
	                help='enables CUDA training')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
	                help='number of epochs to train (default: 10)')
	parser.add_argument('--batch_size', type=int, default=64, metavar='N',
	                help='input batch size for training (default: 64)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
	                help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                help='SGD momentum (default: 0.5)')
	parser.add_argument('--margin', type=float, default=1.0, metavar='M',
	                help='margin for triplet loss (default: 1.0)')
	parser.add_argument('--ckp', default=None, type=str,
	                help='path to load checkpoint')

	parser.add_argument('--dataset', type=str, default='mnist', metavar='M',
	                help='number of training samples (default: 3000)')
	
	parser.add_argument('--num_train_samples', type=int, default=50000, metavar='M',
	                help='number of training samples (default: 3000)')
	parser.add_argument('--num_test_samples', type=int, default=10000, metavar='M',
	                help='number of test samples (default: 1000)')	

	parser.add_argument('--train_log_step', type=int, default=100, metavar='M',
	                help='Number of iterations after which to log the loss')

	global args
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()
	cfg_from_file("config/test.yaml")
	main()