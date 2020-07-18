import random
import numpy as np


class MNIST_DS(object):

    def __init__(self, train_dataset, test_dataset):
        self.__train_labels_idx_map = {}
        self.__test_labels_idx_map = {}

        self.__train_data = train_dataset.data
        self.__test_data = test_dataset.data
        self.__train_labels = train_dataset.targets
        self.__test_labels = test_dataset.targets

        self.__train_labels_np = self.__train_labels.numpy()
        self.__train_unique_labels = np.unique(self.__train_labels_np)

        self.__test_labels_np = self.__test_labels.numpy()
        self.__test_unique_labels = np.unique(self.__test_labels_np)

    def load(self):
        self.__train_labels_idx_map = {}
        for label in self.__train_unique_labels:
            self.__train_labels_idx_map[label] = np.where(self.__train_labels_np == label)[0]

        self.__test_labels_idx_map = {}
        for label in self.__test_unique_labels:
            self.__test_labels_idx_map[label] = np.where(self.__test_labels_np == label)[0]

    def getTriplet(self, split="train"):
        pos_label = 0
        neg_label = 0
        label_idx_map = None
        data = None

        if split == 'train':
            pos_label = self.__train_unique_labels[random.randint(0, len(self.__train_unique_labels) - 1)]
            neg_label = pos_label
            while neg_label is pos_label:
                neg_label = self.__train_unique_labels[random.randint(0, len(self.__train_unique_labels) - 1)]
            label_idx_map = self.__train_labels_idx_map
            data = self.__train_data
        else:
            pos_label = self.__test_unique_labels[random.randint(0, len(self.__test_unique_labels) - 1)]
            neg_label = pos_label
            while neg_label is pos_label:
                neg_label = self.__test_unique_labels[random.randint(0, len(self.__test_unique_labels) - 1)]
            label_idx_map = self.__test_labels_idx_map
            data = self.__test_data

        pos_label_idx_map = label_idx_map[pos_label]
        pos_img_anchor_idx = pos_label_idx_map[random.randint(0, len(pos_label_idx_map) - 1)]
        pos_img_idx = pos_img_anchor_idx
        while pos_img_idx is pos_img_anchor_idx:
            pos_img_idx = pos_label_idx_map[random.randint(0, len(pos_label_idx_map) - 1)]

        neg_label_idx_map = label_idx_map[neg_label]
        neg_img_idx = neg_label_idx_map[random.randint(0, len(neg_label_idx_map) - 1)]

        pos_anchor_img = data[pos_img_anchor_idx].numpy()
        pos_img = data[pos_img_idx].numpy()
        neg_img = data[neg_img_idx].numpy()

        return pos_anchor_img, pos_img, neg_img
