import os
import random
import json
from base_config import cfg

import cv2

class S2S(object):

    def __init__(self):
    	self.__base_path = ""
        self.__skus = ['bags', 'belts', 'dresses', 'eyewear', 'footwear', 'hats', 'leggings', 'outerwear', 'pants', 'skirts', 'tops']

    def load(self):
    	self.__imgs_dict = {}
    	self.__base_path = cfg.DATASETS.S2S.HOME

        self.__train_pairs = {}
        self.__test_pairs = {}
        self.__retrieval_dict = {}

        total_train_pairs = 0
        total_test_pairs = 0

        for sku in self.__skus:
            train_json = os.path.join(self.__base_path, "meta", "meta", "json", "train_pairs_"+sku+".json")
            with open(train_json) as json_file:  
                train_pairs = json.load(json_file)
                total_train_pairs += len(train_pairs)
                self.__train_pairs[sku] = train_pairs

            test_json = os.path.join(self.__base_path, "meta", "meta", "json", "test_pairs_"+sku+".json")
            with open(test_json) as json_file:  
                test_pairs = json.load(json_file)
                total_test_pairs += len(test_pairs)
                self.__test_pairs[sku] = test_pairs

            retrieval_json = os.path.join(self.__base_path, "meta", "meta", "json", "retrieval_"+sku+".json")
            with open(retrieval_json) as json_file:  
                sku_retrieval_pairs = json.load(json_file)
            self.__retrieval_dict[sku] = {}
            for pair in sku_retrieval_pairs:
                self.__retrieval_dict[sku][pair['product']] = pair['photo']

        return total_train_pairs, total_test_pairs

    def getTriplet(self, sku=None, mode="train"):
        if sku is None:
            num_skus = len(self.__skus)
            sku_idx = random.randint(0, num_skus-1)
            sku = self.__skus[sku_idx]
        if mode is 'train':
            pairs = self.__train_pairs[sku]
        else:
            pairs = self.__test_pairs[sku]
        pair_len = len(pairs)
        rdict = self.__retrieval_dict[sku]

        product_img_name = ""
        neg_img_name = ""
        pos_img_name = ""

        while True:
            pair_idx = random.randint(0, pair_len-1)
            train_pair = pairs[pair_idx]
            product_img_name = rdict[train_pair['product']]
            pos_img_name = train_pair['photo']
            pos_img_bbox = train_pair['bbox']
            if self.check_if_img_exists(product_img_name) and self.check_if_img_exists(pos_img_name):
                break

        while True:
            neg_pair_idx = random.randint(0, pair_len-1)
            neg_pair = pairs[neg_pair_idx]
            neg_product_img_name = neg_pair['product']
            if neg_product_img_name is not product_img_name:
                neg_img_name = neg_pair['photo']
                neg_img_bbox = neg_pair['bbox']
                if self.check_if_img_exists(neg_img_name):
                    break

        product_img_name = str(product_img_name).zfill(9)
        pos_img_name = str(pos_img_name).zfill(9)
        neg_img_name = str(neg_img_name).zfill(9)
        product_img_name = os.path.join(self.__base_path, "JPEGImages", str(product_img_name) + ".jpg")
        pos_img_name = os.path.join(self.__base_path, "JPEGImages", str(pos_img_name) + ".jpg")
        neg_img_name = os.path.join(self.__base_path, "JPEGImages", str(neg_img_name) + ".jpg")

        return sku, (product_img_name, None), (pos_img_name, pos_img_bbox), (neg_img_name, neg_img_bbox)

    def check_if_img_exists(self, img_name):
        img_name = str(img_name).zfill(9)
        return os.path.isfile(os.path.join(self.__base_path, "JPEGImages", str(img_name) + ".jpg"))