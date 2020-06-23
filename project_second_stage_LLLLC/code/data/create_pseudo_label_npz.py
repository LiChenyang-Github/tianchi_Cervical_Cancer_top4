"""
    @lichenyang 2019.12.30

    Save pred bbox of test set into npz to be used as pseudo label. 
"""


import numpy as np
import cv2
import os
import os.path as osp
import json
import random
import pdb

from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter

import kfbReader



def get_cls_types():

    CELL_TYPES = (
        'ASC-H',    
        'ASC-US',   
        'HSIL',     
        'LSIL',     
        'Candida',  
        'Trichomonas',  
        )

    return CELL_TYPES


def gen_pseudo_label(pred_json_root, dst_npz_root, \
    keep_score_thred=0.9, kfb_root="/home/admin/jupyter/Data/test/"):

    count = 0

    json_names = os.listdir(pred_json_root)

    CELL_TYPES = get_cls_types()

    for json_name in tqdm(json_names):

        json_dir = osp.join(pred_json_root, json_name)
        kfb_dir = osp.join(kfb_root, '{}.kfb'.format(json_name.split('.')[0]))

        with open(json_dir, 'r') as f:
            json_infos = json.load(f)

        r = kfbReader.reader()
        r.ReadInfo(kfb_dir, 20, True)  ### 20

        for i, json_info in enumerate(json_infos):

            x = json_info['x']
            y = json_info['y']
            w = json_info['w']
            h = json_info['h']
            p = json_info['p']
            c = json_info['class']

            if p > keep_score_thred:

                count += 1

                npz_dir = osp.join(dst_npz_root, c, \
                    '{}_{}.npz'.format(json_name.split('.')[0], i))

                if not osp.exists(osp.dirname(npz_dir)):
                    os.makedirs(osp.dirname(npz_dir))


                img = r.ReadRoi(x, y, w, h, 20).copy()
                label = CELL_TYPES.index(c) + 1 # 1-based

                np.savez_compressed(npz_dir, img=img, label=label)

                # print(w, h, img.shape, label)



    # pdb.set_trace()



def all_score_smaller_than_thred(img_id, score_dict, score_thred=0.8):

    for k in score_dict.keys():
        if img_id in k:
            if score_dict[k] > score_thred:
                return False

    return True




def gen_npz_path(pred_json_root, dst_npz_root, \
    dst_npz_dir="/home/admin/jupyter/Datasets/pseudo_label/pseudo_label_dict_candida.npz", \
    score_threds=(0.6, 0.3, 0.1)):
    
    if not osp.exists(osp.dirname(dst_npz_dir)):
        os.makedirs(osp.dirname(dst_npz_dir))


    count = 0

    res = defaultdict(list)
    res_npz_name_dict = defaultdict(list)
    ignore_dict = defaultdict(list)

    score_dict = {}

    json_names = os.listdir(pred_json_root)

    CELL_TYPES = get_cls_types()

    # for json_name in tqdm(json_names):
    for json_name in json_names:


        json_dir = osp.join(pred_json_root, json_name)

        with open(json_dir, 'r') as f:
            json_infos = json.load(f)


        for i, json_info in enumerate(json_infos):

            x = json_info['x']
            y = json_info['y']
            w = json_info['w']
            h = json_info['h']
            p = json_info['p']
            c = json_info['class']

            for score_thred in score_threds:

                npz_name = '{}_{}.npz'.format(json_name.split('.')[0], i)

                if p > score_thred:

                    res_npz_name_dict[str(score_thred)].append(npz_name)
                    score_dict[npz_name] = p

    for k in res_npz_name_dict.keys():

        counter = Counter([x.split('_')[0] for x in res_npz_name_dict[k]])
        counter_dict = dict(counter)

        for img_id in counter_dict.keys():
            if counter_dict[img_id] < 3:
                if all_score_smaller_than_thred(img_id, score_dict):
                    ignore_dict[k].append(img_id)


    for k in res_npz_name_dict.keys():

        for npz_name in res_npz_name_dict[k]:

            img_id = npz_name.split('_')[0]

            if img_id in ignore_dict[k]:
                continue

            res[k].append(osp.join(dst_npz_root, 'Candida', npz_name))



    pdb.set_trace()

    # np.savez_compressed(dst_npz_dir, npz_dirs=res)

    # pdb.set_trace()





    






if __name__ == '__main__':

    pred_json_root = "/home/admin/jupyter/Projects/mmdetection/output/prediction/pred_json/Candida/r50_3000-2000_multi_scale8-16-32-64_ep20-25-30_combined"
    # pred_json_root = "/home/admin/jupyter/Projects/mmdetection/output/prediction/pred_json/Candida/r50_3000-2000_multi_scale8-16-32-64_ep20-25-30_combined_and_imgsize_as_cropsize"

    dst_npz_root = "/home/admin/jupyter/Datasets/crop_bbox_test"
    keep_score_thred = 0.1

    if not osp.exists(dst_npz_root):
        os.makedirs(dst_npz_root)

    # gen_pseudo_label(pred_json_root, dst_npz_root, keep_score_thred)

    gen_npz_path(pred_json_root, dst_npz_root)







