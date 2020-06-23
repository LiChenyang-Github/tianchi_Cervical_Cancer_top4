"""
    @lichenyang 2019.11.30

    Crop the roi area and save as .npz with update json.
"""

import numpy as np
import cv2
import os
import os.path as osp
import json
import time
import random
import shutil
import pdb

from glob import glob
from tqdm import tqdm

import kfbReader




def save_roi_to_npz(src_dir, dst_dir, update_dir, cell_types):
    json_paths = glob(os.path.join(src_dir, "*.json"))
    update_json_names = os.listdir(update_dir)

    if not osp.isdir(dst_dir):
        os.makedirs(dst_dir)

    # pdb.set_trace()

    for json_path in tqdm(json_paths):

        # use update json
        if osp.basename(json_path) in update_json_names:
            json_path = osp.join(update_dir, osp.basename(json_path))

        # if osp.basename(json_path) in ['2393.json', '8484.json']:
        #     json_path = osp.join(update_dir, osp.basename(json_path))
        # else: 
        #     continue

        filename = json_path.split("/")[-1].split('.')[0]
        pos_path = osp.join(src_dir, filename+'.kfb')
        with open(json_path, 'r') as f:
            json_infos = json.loads(f.read())

        r = kfbReader.reader()
        r.ReadInfo(pos_path, 20, True)  ### 20

        # select the roi coord.
        roi_coords = []
        for json_info in json_infos:
            if json_info['class'] == 'roi':
                coord = {'x': json_info['x'], 'y': json_info['y'], 'w': json_info['w'], 'h': json_info['h']}
                roi_coords.append(coord)

        # print(len(roi_coords))

        roi_cnt = 1
        for roi_coord in roi_coords:
            X, Y, W, H = roi_coord['x'], roi_coord['y'], roi_coord['w'], roi_coord['h']
            img = r.ReadRoi(X, Y, W, H, 20).copy()
            label = np.zeros((0, 5), dtype="int")

            pos_cnt = 0
            for json_info in json_infos:
                if json_info['class'] in cell_types:
                    x, y, w, h = json_info['x'], json_info['y'], json_info['w'], json_info['h']
                    if X < x < X + W and Y < y < Y + H:
                        pos_cnt += 1
                        box = np.zeros((1, 5), dtype="int")
                        box[0, 0] = max(int(x - X), 0)
                        box[0, 1] = max(int(y - Y), 0)
                        box[0, 2] = min(int(x - X + w), W)
                        box[0, 3] = min(int(y - Y + h), H)
                        box[0, 4] = cell_types.index(json_info['class'])
                        # print(json_info['class'], cell_types.index(json_info['class']))
                        if int(x - X + w) > W or int(y - Y + h) > H: print(json_info)
                        label = np.append(label, box, axis=0)

            if pos_cnt == 0:
                continue

            save_path = osp.join(dst_dir, filename + "_" + str(roi_cnt) + ".npz")
            np.savez_compressed(save_path, img=img, label=label)

            roi_cnt += 1
        # print("Finish: ", filename, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



if __name__ == "__main__":
    SRC_DIR = "/home/admin/jupyter/Data/train"
    DST_DIR = "/home/admin/jupyter/Datasets/train_rois_update"
    UPDATE_DIR = "/home/admin/jupyter/Projects/mmdetection/data/update/update_train"
    # UPDATE_DIR = "/home/admin/jupyter/Projects/mmdetection/data/update_20191216"


    CELL_TYPES = (
        'ASC-H',    # 6194
        'ASC-US',   # 5945
        'HSIL',     # 2818
        'LSIL',     # 3382
        'Candida',  # 1651
        'Trichomonas',  # 11510
        )
    save_roi_to_npz(SRC_DIR, DST_DIR, UPDATE_DIR, CELL_TYPES)

