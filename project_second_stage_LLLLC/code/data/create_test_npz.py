import cv2 as cv
import sys
import os.path as osp
import os
sys.path.append(r"/home/admin/jupyter/Downloads/kfbreader_linux")
import kfbReader as kr
from glob import glob
import matplotlib.pyplot as plt
import json
from multiprocessing import Pool
import numpy as np


#
# cut neg_sample roi center
#
# 读取json文件
test_kfb_dir = r"/home/admin/jupyter/Data/test"
train_kfb_dir = r"/home/admin/jupyter/Data/train"
train_json_list = glob(osp.join(train_kfb_dir,"*.json"))
test_json_list = glob(osp.join(test_kfb_dir,"*.json"))
print(len(train_json_list))
print(len(test_json_list))


save_test_dir = r"/home/admin/jupyter/Data/test_roi_npz"
if not osp.isdir(save_test_dir):
    os.makedirs(save_test_dir)

def load_json(json_path):
    with open(json_path,"r") as f:
        js = json.load(f)
    return js

# 保存test_roi至指定文件夹，文件名为 id_cnt_topleftx_toplefty_buttomrightx_buttomrighty_Width_Height.npz
def cut_save_test_roi(kfb_path,json_path,save_dir):
    kfb_reader = kr.reader()
    kr.reader.ReadInfo(kfb_reader,kfb_path,20,False)
    
    test_roi_list = load_json(json_path)
    basename = osp.splitext(osp.basename(json_path))[0]
    for i,test_roi in enumerate(test_roi_list):
#         print(test_roi)
        x,y,w,h = test_roi['x'],test_roi['y'],test_roi['w'],test_roi['h']
        npz_name = osp.join(save_dir,basename+"_"+str(i)+".npz")
        roi_image = kfb_reader.ReadRoi(x,y,w,h,20)
        np.savez_compressed(npz_name,img=roi_image,label=(np.array([x,y,x+w,y+h])[None,:]))

# kfb 中 roi 个数与类别是否具有关系？？
# temp_list = []
# for i in range(len(test_json_list)):
#     temp_list.append(len(load_json(test_json_list[i])))
# plt.hist(temp_list)

P = Pool(8)
for i in range(len(test_json_list)):
    test_json_path = test_json_list[i]
    test_kfb_path = test_json_path.replace(".json",".kfb")
    P.apply_async(cut_save_test_roi,(test_kfb_path,test_json_path,save_test_dir))


#
# cut neg_sample roi center
#
neg_save_dir = r"/home/admin/jupyter/Datasets/neg_rois"
if not osp.isdir(neg_save_dir):
    os.makedirs(neg_save_dir)
    
set_json = set([x.replace(".json",'.kfb') for x in glob(r"/home/admin/jupyter/Data/train/*.json")])
set_kfb = set(glob(r"/home/admin/jupyter/Data/train/*.kfb"))
set_neg = set_kfb-set_json

def randomcrop_neg_kfb(kfb_path,save_dir,target_size=(4000,4000)):
    kfb_reader = kr.reader()
    kr.reader.ReadInfo(kfb_reader,kfb_path,20,False)
    Width,Height = kfb_reader.getWidth(),kfb_reader.getHeight()
    center_x,center_y = Width//2,Height//2
    target_w,target_h = target_size
    basename = osp.splitext(osp.basename(kfb_path))[0]
    
    label = (np.array([0,0,4000,4000,-1]))[None,:]
    # center_crop
    offset_x = np.random.randint(center_x-target_w//2,center_x)
    offset_y = np.random.randint(center_y-target_h//2,center_y)
    crop_image = kfb_reader.ReadRoi(offset_x,offset_y,target_w,target_h,20)
    
    npz_name = osp.join(save_dir,basename+"_"+"center_crop"+".npz")
    np.savez_compressed(npz_name,img=crop_image,label=label)
    
    #topleft crop
    offect_x = np.random.randint(max(0,center_x-target_w*2),center_x-target_w)
    offect_y = np.random.randint(max(0,center_y-target_h*2),center_y-target_h)
    crop_image = kfb_reader.ReadRoi(offect_x,offect_y,target_w,target_h,20)
    
    npz_name = osp.join(save_dir,basename+"_"+"top_left_crop"+".npz")
    np.savez_compressed(npz_name,img=crop_image,label=label)
    
    #topright crop
    offect_x = np.random.randint(center_x+target_w,center_x+2*target_w)
    offect_y = np.random.randint(center_y-target_h*2,center_y-target_h)
    crop_image = kfb_reader.ReadRoi(offect_x,offect_y,target_w,target_h,20)
    
    npz_name = osp.join(save_dir,basename+"_"+"top_right_crop"+".npz")
    np.savez_compressed(npz_name,img=crop_image,label=label)
    
    #buttomleft crop
    offect_x = np.random.randint(center_x-target_w*2,center_x-target_w)
    offect_y = np.random.randint(center_y+target_h,center_y+target_h*2)
    crop_image = kfb_reader.ReadRoi(offect_x,offect_y,target_w,target_h,20)
    
    npz_name = osp.join(save_dir,basename+"_"+"buttom_left_crop"+".npz")
    np.savez_compressed(npz_name,img=crop_image,label=label)
    
    #buttomright crop
    offect_x = np.random.randint(center_x+target_w,center_x+2*target_w)
    offect_y = np.random.randint(center_y+target_h,center_y+2*target_h)
    crop_image = kfb_reader.ReadRoi(offect_x,offect_y,target_w,target_h,20)
    
    npz_name = osp.join(save_dir,basename+"_"+"buttom_right_crop"+".npz")
    np.savez_compressed(npz_name,img=crop_image,label=label)
P = Pool(8)
for neg_kfb in set_neg:
    P.apply_async(randomcrop_neg_kfb,(neg_kfb,neg_save_dir))
    
    
# 裁剪train box 直接保存
from tqdm import tqdm

save_dir = r"/home/admin/jupyter/Datasets/crop_bbox"
train_rois_update_dir = r"/home/admin/jupyter/Datasets/train_rois_update"
list_train_rois = os.listdir(train_rois_update_dir)


def cut_image_label(npz_path):
    npz = np.load(npz_path)
    image,label = npz['img'],npz['label']
    basename = os.path.basename(npz_path)
    basename = basename.split(".")[0]
    for i in range(len(label)):
        x1,y1,x2,y2,label_gt = label[i,:]
        crop_image = image[y1:y2,x1:x2,:]
        
        if label_gt == 4:#candida
            save_path = save_dir + r"/Candida"
        elif label_gt == 5:
            save_path = save_dir + r"/Trichomonas"
        else:
            save_path = save_dir + r"/Pos"
        save_path = os.path.join(save_path,basename+"_"+str(i))
        np.savez_compressed(save_path,img=crop_image,label=label_gt+1)


for file in tqdm(list_train_rois):
    npz_path = os.path.join(train_rois_update_dir,file)
    cut_image_label(npz_path)


from glob import glob
candict_list = glob(r"/home/admin/jupyter/Datasets/crop_bbox/Candida/*.npz")
pos_list = glob(r"/home/admin/jupyter/Datasets/crop_bbox/Pos/*.npz")
trichomonas_list = glob(r"/home/admin/jupyter/Datasets/crop_bbox/Trichomonas/*.npz")
np.savez_compressed("copy_pasting_dict.npz",pos=pos_list,candida=candict_list,trichomonas=trichomonas_list)