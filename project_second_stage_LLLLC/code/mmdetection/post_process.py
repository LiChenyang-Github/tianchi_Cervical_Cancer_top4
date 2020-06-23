import numpy as np
import cv2
from glob import glob
import os
from collections import Counter
import matplotlib.pyplot as plt
import json
from collections import Counter
import cv2 as cv
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def load_json(json_path):
    basename = os.path.basename(json_path)
    with open(json_path,"r") as f:
        js = json.load(f)
    return {basename:js}


def convert_ListDict_to_ListNp(pred_json_list):
    type_list = ["ASC-H","ASC-US","HSIL","LSIL","Candida","Trichomonas"]
    list_np = {}
    for key in pred_json_list.keys():
        pred_json = pred_json_list[key]
        temp = []
        for pred in pred_json:
#             temp.append([pred['x'],pred['y'],pred['w'],pred['h'],pred['p'],type_list.index(pred['class'])])
            temp.append([pred['x'],pred['y'],pred['w'],pred['h'],pred['p'],pred['class']])
        list_np.update({key:np.array(temp)})
    return list_np

def filter_w_h_threshold(pred_json_ListNp,threshold=10):
    for key in pred_json_ListNp.keys():
        if pred_json_ListNp[key].ndim==2:
            w_indice = (pred_json_ListNp[key][:,2]).astype(int)>threshold
            pred_json_ListNp[key] = pred_json_ListNp[key][w_indice,:]
            h_indice = (pred_json_ListNp[key][:,3]).astype(int)>threshold
            pred_json_ListNp[key] = pred_json_ListNp[key][h_indice,:]
    return pred_json_ListNp

def calc_len(pred_json_ListNp):
    cnt = 0
    for key in pred_json_ListNp.keys():
        cnt += len(pred_json_ListNp[key])
    return cnt

def filter_num_tri(pred_json_ListNp,threshlod=2):
    # 如果pred_json中Tri预测个数小于2，则较大概率可以排除
    for key in pred_json_ListNp.keys():
        pred_json = pred_json_ListNp[key]
        if pred_json.ndim==2:
            tri_indice = (pred_json[:,-1]=="Trichomonas")
            if np.sum(tri_indice) <=threshlod:
                pred_json_ListNp[key] = pred_json_ListNp[key][pred_json[:,-1]!="Trichomonas"]
    return pred_json_ListNp

def filter_neg_pos(pred_json_ListNp,num_tri=100,percents=0.95):
    for key in pred_json_ListNp.keys():
        pred_json = pred_json_ListNp[key]
        if pred_json.ndim==2:
            counter = dict(Counter(pred_json[:,-1]))
            if "Trichomonas" in counter:
                num_Trichomonas = counter.pop("Trichomonas")
                num_not_Trichomonas = sum(counter.values())
    #             print(num_Trichomonas,num_not_Trichomonas)
                if num_not_Trichomonas and (num_Trichomonas/(num_not_Trichomonas+num_Trichomonas) > percents):
                    pred_json_ListNp[key] = pred_json[pred_json[:,-1]=="Trichomonas"]
    return pred_json_ListNp

def filter_NotCandida_threshold(pred_json_ListNp,threshold=0.8,num_most_p=6,num_key=6):
    cnt = 0
    dict_cnt = {}
    for key in pred_json_ListNp.keys():
        mean_p_list = []
        pred_json = pred_json_ListNp[key]
        if pred_json.ndim==2:
            counter = Counter(pred_json[:,-1])
            num_counter = min([num_key,len(counter)])
            most_key = counter.most_common(num_counter) # 预测最多的num_key个类别
            most_key_list = [item[0] for item in most_key]

            for key_ in most_key_list:
                pred_key_p = (pred_json[pred_json[:,-1]==key_])[:,-2].astype(np.float)
                if len(pred_key_p) <= num_most_p:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p)))
                else:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p[pred_key_p.argsort()[::-1][0:num_most_p]])))
                    
        for item in mean_p_list:
            if item[0] == "Candida" and item[2]>threshold:
#                 print(key,mean_p_list)
                pred_json[pred_json[:,-1]!="Candida",-2] = str(0.001)
#                 print(pred_json)
                pred_json_ListNp[key] = pred_json
                
                for _ in mean_p_list:
                    if _[0] not in dict_cnt:
                        dict_cnt[_[0]] = _[1]
                    else:
                        dict_cnt[_[0]] += _[1]
                break
    dict_cnt.pop("Candida")
    print(dict_cnt)
    print(np.sum(list(dict_cnt.values())))
    return pred_json_ListNp


def filter_NotPos_threshold(pred_json_ListNp,threshold=0.8,num_most_p=6,num_key=6):
    cnt = 0
    dict_cnt = {}
    for key in pred_json_ListNp.keys():
        mean_p_list = []
        pred_json = pred_json_ListNp[key]
        if pred_json.ndim==2:
            counter = Counter(pred_json[:,-1])
            num_counter = min([num_key,len(counter)])
            most_key = counter.most_common(num_counter) # 预测最多的num_key个类别
            most_key_list = [item[0] for item in most_key]

            for key_ in most_key_list:
                pred_key_p = (pred_json[pred_json[:,-1]==key_])[:,-2].astype(np.float)
                if len(pred_key_p) <= num_most_p:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p)))
                else:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p[pred_key_p.argsort()[::-1][0:num_most_p]])))
        flag = True
        for item in mean_p_list:
            if item[2]>0.50 and item[0]=="Candida":
                flag = False
        if flag:
            for item in mean_p_list:
                if item[2]>threshold and (item[0]!="Candida" and item[0]!="Trichomonas"):
                    pred_json[pred_json[:,-1]=="Candida",-2] = str(0.001)
                    pred_json[pred_json[:,-1]=="Trichomonas",-2] = str(0.001)
    #                 print(pred_json)
                    pred_json_ListNp[key] = pred_json

                    for _ in mean_p_list:
                        if _[0] not in dict_cnt:
                            dict_cnt[_[0]] = _[1]
                        else:
                            dict_cnt[_[0]] += _[1]
                    break
    

    print(dict_cnt)
    if "Candida" in dict_cnt:
        print(dict_cnt.pop("Candida")+dict_cnt.pop("Trichomonas"))
    return pred_json_ListNp
    
def analyse(pred_json_ListNp,num_key=6,num_most_p=3):
    cnt = 0
    dict_cnt = {}
    for key in pred_json_ListNp.keys():
        mean_p_list = []
        pred_json = pred_json_ListNp[key]
        if pred_json.ndim==2:
            counter = Counter(pred_json[:,-1])
            num_counter = min([num_key,len(counter)])
            most_key = counter.most_common(num_counter) # 预测最多的num_key个类别
            most_key_list = [item[0] for item in most_key]



            for key_ in most_key_list:
                pred_key_p = (pred_json[pred_json[:,-1]==key_])[:,-2].astype(np.float)
                if len(pred_key_p) <= num_most_p:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p)))
                else:
                    mean_p_list.append((key_,len(pred_key_p),np.mean(pred_key_p[pred_key_p.argsort()[::-1][0:num_most_p]])))
            for item in mean_p_list:
                if item[2] > 0.85 and item[0]=="Candida":
                    print(key,mean_p_list)
                    break


    
def _convert_ListDict_ListNp(ListDict):
    ListNp = []
    for item in ListDict:
        ListNp.append([item['x'],item['y'],item['x']+item['w'],item['y']+item['h']])
    ListNp = np.array(ListNp)
    return ListNp

def load_js(js_path):
    with open(js_path,"r") as f:
        return json.load(f)
    
def filter_outer(pred_json_ListNp,test_json_dir):
    temp_cnt = 0
    for key in pred_json_ListNp.keys():
        idx = key.split('.')[0]
        idx = str(idx)+".json"
        pred_json = pred_json_ListNp[idx]
        gt_roi_label_list = load_js(os.path.join(test_json_dir,idx))
        gt_roi_label_Np = _convert_ListDict_ListNp(gt_roi_label_list)
        
        save_list = []
        for i in range(len(pred_json)): #外循环  遍历所有box
            x1,y1,w,h = pred_json[i,:4].astype(int)
            x2,y2 = x1+w,y1+h
            pred_p,pred_class = float(pred_json[i,4]),pred_json[i,-1]
            for j in range(len(gt_roi_label_Np)): # 内循环 遍历所有roi_gt
                roi_label = gt_roi_label_Np[j]
                left,top,right,buttom = roi_label
                
                # 完全出界就过滤，（x1,y1）在框内则修正w,h
                if (left<=x1) and (x1<=right) and (top<=y1) and (y1<=buttom):
                    if (left<=x2) and (x2<=right) and (top<=y2) and (y2<=buttom):
                        save_list.append([x1,y1,w,h,pred_p,pred_class])
                    else:
                        if x2 > right:
                            w = right - x1
                        if y2 > buttom:
                            h = buttom-y1
                        save_list.append([x1,y1,w,h,pred_p,pred_class])
                        pass
                
#         save_Np = np.concatenate(tuple(save_list),asix=1)
        save_Np = np.array(save_list)
        pred_json_ListNp[key] = save_Np
    return pred_json_ListNp

def save_pred_json_ListNp(pred_json_ListNp,save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for key in pred_json_ListNp.keys():
        pred_json = pred_json_ListNp[key]
        pred_ListDict = []
        for i in range(len(pred_json)):
            x,y,w,h = pred_json[i,:4]
            x,y,w,h = [int(item) for item in [x,y,w,h]]
            pred_p = float(pred_json[i,-2])
            pred_class = pred_json[i,-1]
            pred_ListDict.append({'x':x,'y':y,'w':w,'h':h,'p':pred_p,'class':pred_class})
        with open(os.path.join(save_dir,key),'w') as f:
            json.dump(pred_ListDict,f)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("--pred_json_dir",type=str,help="path of pred_json_dir")
    parser.add_argument("--save_dir",type=str,help="path of save_dir")
    parser.add_argument("--test_json_dir",type=str,default=r"/home/admin/jupyter/Data/test/",help="path of test_json_dir")

    arg = parser.parse_args()
    pred_json_dir = arg.pred_json_dir
    test_json_dir = arg.test_json_dir
    save_dir = arg.save_dir

    json_path  = glob(os.path.join(pred_json_dir,"*.json"))
    pred_json_list = {}
    for pred_json in json_path:
        pred_json_list.update(load_json(pred_json))

    # 总预测方框数目
    total_pred_box_num = 0
    for key in pred_json_list.keys():
        total_pred_box_num += len(pred_json_list[key])
    print("总预测方框数目为　：　",total_pred_box_num)

    w_list,h_list = [],[]
    for key in pred_json_list.keys():
        roi_item = pred_json_list[key]
        for box in roi_item:
            w_list.append(box['w'])
            h_list.append(box['h'])
    print("min(w) : ",min(w_list))
    print("min(h) : ",min(h_list))
    print("max(w) : ",max(w_list))
    print("max(h) : ",max(h_list))

    pred_json_ListNp = convert_ListDict_to_ListNp(pred_json_list)
    print("before post-processing")
    print(calc_len(pred_json_ListNp))

    filter_outer(pred_json_ListNp,test_json_dir)
    print("After filter outer")
    print(calc_len(pred_json_ListNp))

    threshold = 0.85
    num_most_p = 3
    pred_json_ListNp = filter_NotCandida_threshold(pred_json_ListNp,threshold=threshold,num_most_p=num_most_p)
    print("after filter_NotCandida with threshold={},num_most_p={} ".format(threshold,num_most_p))
    print(calc_len(pred_json_ListNp))

    save_pred_json_ListNp(pred_json_ListNp,save_dir=save_dir)