import os
import random


class BaseDset(object):

    def __init__(self):
        self.__base_path = ""

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

    def load(self, base_path):
        self.__base_path = base_path
        train_dir = os.path.join(self.__base_path, 'train')
        test_dir = os.path.join(self.__base_path, 'test')

        self.__train_set = {}
        self.__test_set = {}
        self.__train_keys = []
        self.__test_keys = []

        for class_id in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_id)
            self.__train_set[class_id] = []
            self.__train_keys.append(class_id)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.__train_set[class_id].append(img_path)

        for class_id in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_id)
            self.__test_set[class_id] = []
            self.__test_keys.append(class_id)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.__test_set[class_id].append(img_path)

        return len(self.__train_keys), len(self.__test_keys)

    def getTriplet(self, split='train'):
        if split == 'train':
            dataset = self.__train_set
            keys = self.__train_keys
        else:
            dataset = self.__test_set
            keys = self.__test_keys

        pos_idx = 0
        neg_idx = 0
        pos_anchor_img_idx = 0
        pos_img_idx = 0
        neg_img_idx = 0

        pos_idx = random.randint(0, len(keys) - 1)
        while True:
            neg_idx = random.randint(0, len(keys) - 1)
            if pos_idx != neg_idx:
                break

        pos_anchor_img_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
        while True:
            pos_img_idx = random.randint(0, len(dataset[keys[pos_idx]]) - 1)
            if pos_anchor_img_idx != pos_img_idx:
                break

        neg_img_idx = random.randint(0, len(dataset[keys[neg_idx]]) - 1)

        pos_anchor_img = dataset[keys[pos_idx]][pos_anchor_img_idx]
        pos_img = dataset[keys[pos_idx]][pos_img_idx]
        neg_img = dataset[keys[neg_idx]][neg_img_idx]

        return pos_anchor_img, pos_img, neg_img
