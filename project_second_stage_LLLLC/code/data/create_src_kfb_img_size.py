
import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm
from glob import glob



def gen_src_img_size():
    """
        Write the size of src kfb img into a json file.
    """

    # import kfbReader as kr

    kfb_root = "/home/admin/jupyter/Data/test/"
    kfb_dirs = glob(osp.join(kfb_root, '*.kfb'))
    json_dir = "/home/admin/jupyter/Datasets/src_kfb_img_size.json"

    res = {}

    w_max, h_max = 0, 0

    for kfb_dir in tqdm(kfb_dirs):

        img_name = osp.basename(kfb_dir).split('.')[0]

        kfb_reader = kr.reader()
        kr.reader.ReadInfo(kfb_reader, kfbPath=kfb_dir, scale=20, readAll=False)

        Width, Height = kfb_reader.getWidth(), kfb_reader.getHeight()

        res_cur = dict(width=Width, height=Height)

        res[img_name] = res_cur


    pdb.set_trace()

    with open(json_dir, 'w') as f:

        json.dump(res, f)


if __name__ == '__main__':
    gen_src_img_size()