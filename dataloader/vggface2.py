import os
import random
from config.base_config import cfg


class VGGFace2(object):

	def __init__(self):
		self.__base_path = ""

		self.__train_set = {}
		self.__test_set = {}
		self.__train_keys = []
		self.__test_keys = []

	def load(self):
		self.__base_path = cfg.DATASETS.VGGFACE2.HOME
		train_dir = os.path.join(self.__base_path, 'train')
		test_dir = os.path.join(self.__base_path, 'test')

		self.__train_set = {}
		self.__test_set = {}
		self.__train_keys = []
		self.__test_keys = []

		for person_id in os.listdir(train_dir):
			person_dir = os.path.join(train_dir, person_id)
			self.__train_set[person_id] = []
			self.__train_keys.append(person_id)
			for face_id in os.listdir(person_dir):
				face_img_path = os.path.join(person_dir, face_id)
				self.__train_set[person_id].append(face_img_path)

		for person_id in os.listdir(test_dir):
			person_dir = os.path.join(test_dir, person_id)
			self.__test_set[person_id] = []
			self.__test_keys.append(person_id)
			for face_id in os.listdir(person_dir):
				face_img_path = os.path.join(person_dir, face_id)
				self.__test_set[person_id].append(face_img_path)

		return len(self.__train_keys), len(self.__test_keys)

	def getTriplet(self, mode='train'):
		if mode == 'train':
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
