from config.base_config import cfg
from dataloader.base_dset import BaseDset


class VGGFace2(BaseDset):

	def __init__(self):
		super(VGGFace2, self).__init__()

	def load(self):
		base_path = cfg.DATASETS.VGGFACE2.HOME
		super(VGGFace2, self).load(base_path)
