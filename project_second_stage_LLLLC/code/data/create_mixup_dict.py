'''
@Author      : now more
@Connect     : lin.honghui@qq.com
@LastEditors : now more
@Description : 
@LastEditTime: 2020-01-10 22:04:24
'''
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2 as cv
neg_rois_dir = r"/home/admin/jupyter/Datasets/neg_rois"
train_rois_dir = r"/home/admin/jupyter/Datasets/train_rois_update"
import cv2

mix_up_dict = {}
background = glob(os.path.join(neg_rois_dir,"*.npz"))


train_rois_list = glob(os.path.join(train_rois_dir,"*.npz"))
Candida = []
Trichomonas = []
pos = []
for i in range(len(train_rois_list)):
    npz = np.load(train_rois_list[i])
    label = npz['label'][:,-1]

    if 4 in label:
        Candida.append(train_rois_list[i])
    if 5 in label:
        Trichomonas.append(train_rois_list[i])
    if 1 in label or 2 in label or 0 in label or 3 in label:
        pos.append(train_rois_list[i])

np.savez_compressed(r"/home/admin/jupyter/Datasets/mix_up_dict.npz",background=background,Candida=Candida,Trichomonas=Trichomonas,pos=pos)
