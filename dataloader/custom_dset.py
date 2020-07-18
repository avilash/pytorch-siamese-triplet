from config.base_config import cfg
from dataloader.base_dset import BaseDset


class Custom(BaseDset):

    def __init__(self):
        super(Custom, self).__init__()

    def load(self):
        base_path = cfg.DATASETS.CUSTOM.HOME
        super(Custom, self).load(base_path)
