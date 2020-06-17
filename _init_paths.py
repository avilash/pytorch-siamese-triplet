import os.path as osp
import sys


def add_path(res_path):
    if res_path not in sys.path:
        sys.path.insert(0, res_path)


this_dir = osp.dirname(__file__)

add_path(osp.join(this_dir))

path = osp.join(this_dir, 'config/')
add_path(path)

path = osp.join(this_dir, 'dataloader/')
add_path(path)

path = osp.join(this_dir, 'model/')
add_path(path)

path = osp.join(this_dir, 'utils/')
add_path(path)
