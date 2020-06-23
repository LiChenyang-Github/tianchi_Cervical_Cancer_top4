"""
    @lichenyang 2019.12.27

    Load the json floder and do post processing, finally save the results into another json folder.

"""

import os
import os.path as osp
import json
import time
import pdb
import shutil

import argparse

import numpy as np

from tqdm import tqdm
from collections import defaultdict



COUNTER = 0

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Json post processing.')

  parser.add_argument('--src_json_roots', dest='src_json_roots',
                      help='the json directories to be processed.',
                      nargs='+', type=str, default='')
  parser.add_argument('--dst_json_root', dest='dst_json_root',
                      help='directory to the res json.',
                      default="./")


  # 1. Select the res from different json folder for each class.
  parser.add_argument('--select_res_from_diff_json', dest='select_res_from_diff_json',
                      help='select the res from different json folder for each class.',
                      action='store_true')
  parser.add_argument('--select_method', dest='select_method',
                      help='the method to do selection.',
                      type=str, default='[]')

  # 2. Cross class suppression.
  parser.add_argument('--cross_class_suppression', dest='cross_class_suppression',
                      help='do cross class suppression.',
                      action='store_true')
  parser.add_argument('--cross_class_id', dest='cross_class_id',
                      help='indicate the class to do cross suppression.',
                      type=str, default='()')
  parser.add_argument('--cross_cls_sup_thred', dest='cross_cls_sup_thred',
                      help='the iou threshold to do cross class suppression.',
                      default=0.8, type=float)

  # 3. Class-wise bbox voting.
  parser.add_argument('--cls_wise_bbox_voting', dest='cls_wise_bbox_voting',
                      help='do class-wise bbox voting.',
                      action='store_true')
  parser.add_argument('--cls_wise_bbox_voting_thred', dest='cls_wise_bbox_voting_thred',
                      help='indicate the bbox voting thred for each class.',
                      type=str, default='()')



  args = parser.parse_args()
  return args


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



def one_cls_vote(dets, vote_iou_thred, fuse_method):

  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 5]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  # ids = np.where(areas == 0)[0]
  # if ids.shape[0] > 0:
  #   print(x1[ids], x2[ids], y1[ids], y2[ids])

  # keep = []
  cur_bbox = []
  res_bbox = []
  while order.size > 0:
      i = order.item(0)
      # keep.append(i)
      cur_bbox.append(dets[i, :])

      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])

      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h

      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      inds_keep = np.where(ovr <= vote_iou_thred)[0]
      inds_vote = np.where(ovr > vote_iou_thred)[0]
      # print("inds_vote", inds_vote.shape[0])

      for j in range(inds_vote.shape[0]):
        cur_bbox.append(dets[order[inds_vote[j] + 1], :])

      # if len(cur_bbox) > 1:
      #   print("len(cur_bbox) larger than 1.")
      #   raise
      res_bbox.append(bbox_voting(cur_bbox, fuse_method))
      cur_bbox = []
      order = order[inds_keep + 1]

  res_bbox = np.array(res_bbox)

  return res_bbox


def bbox_voting(bbox_list, fuse_method):
  bbox_num = len(bbox_list)
  if bbox_num == 1:
    return bbox_list[0]

  bbox_arr = np.array(bbox_list)
  score_sum = np.sum(bbox_arr[:, 5])
  score_max = np.max(bbox_arr[:, 5])

  res_vote = np.zeros(6)

  w = bbox_arr[:, 5] / score_sum

  res = bbox_arr[:, :4] * w[:, np.newaxis]
  # print(res.shape)

  res_vote[:4] = np.sum(res, axis=0)
  res_vote[4] = bbox_list[0][4]
  if fuse_method == 'vote_max':
    res_vote[5] = score_max
  elif fuse_method == 'vote_avg':
    res_vote[5] = score_sum / bbox_num

  # print(bbox_arr, res_vote)


  return res_vote




def multi_cls_vote(dets, thred=0.3, fuse_method='vote_max'):
  """
    Do multi cls voting.
    - inputs:
      dets: np.array(N, 6), (x1, y1, x2, y2, cls_id, score)
    - outputs:
      res: np.array(N, 6), (x1, y1, x2, y2, cls_id, score)
  """

  cls_types = get_cls_types()
  cls_num = len(cls_types)
  res = []

  suppress_num = {}

  for i in range(cls_num):
    cur_cls_id = np.where(dets[:, 4] == i+1)[0]
    if len(cur_cls_id) == 0:
      suppress_num[cls_types[i]] = 0
      continue
    if isinstance(thred, tuple):
      cur_clc_det = one_cls_vote(dets[cur_cls_id, :], thred[i], fuse_method)
    else:
      cur_clc_det = one_cls_vote(dets[cur_cls_id, :], thred, fuse_method)

    res.append(cur_clc_det)

    suppress_num[cls_types[i]] = len(cur_cls_id) - len(cur_clc_det)


  if len(res) == 0:
    return np.zeros((0, 6)), suppress_num
  else:
    res = np.vstack(res)
    return res, suppress_num



def dump_to_json(res, json_root):
  """
    Dump the final pred res into json files.
    -inputs:
      res: dict. 
        key: json_name.
        value: list of final json style.
    - json_root: the dir of json folder to save the res.
  """


  for k in res.keys():

    json_dir = osp.join(json_root, k)
    with open(json_dir, 'w') as f:
      json.dump(res[k], f)


def select_res_from_jsons(src_json_roots, select_method):

  res = defaultdict(list)

  CELL_TYPES = get_cls_types()

  for i, src_json_root in enumerate(src_json_roots):

    keep_cls_id = select_method[i]

    cell_types_cur = [CELL_TYPES[x-1] for x in keep_cls_id]

    # print(src_json_root, keep_cls_id, cell_types_cur)

    json_names = os.listdir(src_json_root)

    print("Select {} from {}".format(keep_cls_id, src_json_root))

    for json_name in tqdm(json_names):

      json_dir = osp.join(src_json_root, json_name)

      with open(json_dir, 'r') as f:
        json_infos = json.load(f)

      for json_info in json_infos:

        if json_info['class'] in cell_types_cur:\
          # print(json_info)
          res[json_name].append(json_info)

  return res


def json_list_to_arr(json_list):
  """
    json style list to txt style arr.
      1. xywh --> xyxy
      2. cls_name --> cls_id (1-based)
    - inputs:
      json_list: list(dict)
  """

  bbox_num = len(json_list)
  arr_res = np.zeros((bbox_num, 6))
  CELL_TYPES = get_cls_types()


  for i in range(bbox_num):

    dict_cur = json_list[i]

    x = dict_cur['x']
    y = dict_cur['y']
    w = dict_cur['w']
    h = dict_cur['h']
    p = dict_cur['p']
    cls_id = CELL_TYPES.index(dict_cur['class']) + 1  # 1-based

    arr_res[i, 0] = x
    arr_res[i, 1] = y
    arr_res[i, 2] = x + w - 1
    arr_res[i, 3] = y + h - 1
    arr_res[i, 4] = cls_id
    arr_res[i, 5] = p

  return arr_res


def arr_to_json_list(arr_input):

  res = []
  cls_types = get_cls_types()

  for i in range(arr_input.shape[0]):
    x = int(arr_input[i][0])
    y = int(arr_input[i][1])
    w = int(arr_input[i][2] - arr_input[i][0] + 1)
    h = int(arr_input[i][3] - arr_input[i][1] + 1)
    cls_str = cls_types[int(arr_input[i][4]) - 1]
    score = float(arr_input[i][5])


    dict_cur = {}
    dict_cur['x'] = x
    dict_cur['y'] = y
    dict_cur['w'] = w
    dict_cur['h'] = h
    dict_cur['p'] = score
    dict_cur['class'] = cls_str

    res.append(dict_cur)

  return res

def cross_class_suppress_one_image(arr_input, cross_class_id):

  assert len(cross_class_id) > 1
  # pdb.set_trace()

  if len(arr_input) == 0:
    return np.zeros((0, 6))

  CELL_TYPES = get_cls_types()

  cross_class_list = []
  other_class_list = []

  for i in range(1, len(CELL_TYPES)+1):
    if i in cross_class_id:
      cross_class_list.append(arr_input[np.where(arr_input[:, 4] == i)[0], :])
    else:
      other_class_list.append(arr_input[np.where(arr_input[:, 4] == i)[0], :])


  cross_class_arr = np.vstack(cross_class_list)

  cross_class_arr_sup = cross_class_suppress_core(cross_class_arr)

  # print(other_class_list, cross_class_arr_sup.shape)
  res = np.vstack(other_class_list + [cross_class_arr_sup])

  return res






def cross_class_suppress_core(dets, sup_iou_thred=0.9, sup_coef=0.8):

  global COUNTER

  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 5]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  # ids = np.where(areas == 0)[0]
  # if ids.shape[0] > 0:
  #   print(x1[ids], x2[ids], y1[ids], y2[ids])

  # keep = []
  cur_bbox = []
  res_bbox = []
  while order.size > 0:
      i = order.item(0)
      # keep.append(i)
      cur_bbox.append(dets[i, :])

      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])

      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h

      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      inds_keep = np.where(ovr <= sup_iou_thred)[0]
      inds_sup = np.where(ovr > sup_iou_thred)[0]
      # print("inds_sup", inds_sup.shape[0])

      for j in range(inds_sup.shape[0]):
        COUNTER += 1
        bbox_info = dets[order[inds_sup[j] + 1], :].copy()
        bbox_info[5] = bbox_info[5] * sup_coef
        cur_bbox.append(bbox_info)

        # import pdb
        # pdb.set_trace()

      # if len(cur_bbox) > 1:
      #   print("len(cur_bbox) larger than 1.")
      #   raise
      res_bbox.extend(cur_bbox)
      cur_bbox = []
      order = order[inds_keep + 1]

  res_bbox = np.array(res_bbox)

  if len(res_bbox) == 0:
    res_bbox = np.empty((0, 6)) 

  return res_bbox




def cross_class_suppress(json_style_res, cross_class_id):
  """
    Do cross class suppression.
  """

  print("Do cross class suppression for {}".format(cross_class_id))


  res = {}
  arr_style_res = {}

  for k in tqdm(json_style_res.keys()):
    arr_cur = json_list_to_arr(json_style_res[k])  ###

    # pdb.set_trace()

    arr_style_res[k] = cross_class_suppress_one_image(arr_cur, cross_class_id)

    res[k] = arr_to_json_list(arr_style_res[k])


  return res


def class_wise_bbox_voting(json_style_res, cls_wise_bbox_voting_thred):
  """
    Do class wise bbox voting.
  """

  thred_len = len(cls_wise_bbox_voting_thred)
  assert thred_len == 0 or thred_len == 6

  res = {}

  final_sup_num = defaultdict(int)


  for k in tqdm(json_style_res.keys()):
    arr_cur = json_list_to_arr(json_style_res[k])  ###

    if thred_len == 6:
      arr_voting, sup_num = multi_cls_vote(arr_cur, thred=cls_wise_bbox_voting_thred)
    else:
      arr_voting, sup_num = multi_cls_vote(arr_cur)

    res[k] = arr_to_json_list(arr_voting)

    # pdb.set_trace()

    # for k_sup in sup_num.keys():
    #   final_sup_num[k_sup] += sup_num[k_sup]

  # print(final_sup_num)

  return res


def replace_json(dst_json_root):

  json_1_dir = "./data/update/20200106_updata_json/6198_del_roi.json"
  json_2_dir = "./data/update/20200106_updata_json/9379_del_roi.json"

  dst_json_1_dir = osp.join(dst_json_root, '6198.json')
  dst_json_2_dir = osp.join(dst_json_root, '9379.json')


  shutil.copy(json_1_dir, dst_json_1_dir)
  shutil.copy(json_2_dir, dst_json_2_dir)

  print("Finish replacing {} and {} two jsons.".format('6198', '9379'))






if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  src_json_roots = args.src_json_roots
  dst_json_root = args.dst_json_root
  select_res_from_diff_json = args.select_res_from_diff_json
  cross_class_suppression = args.cross_class_suppression
  cls_wise_bbox_voting = args.cls_wise_bbox_voting
  select_method = eval(args.select_method)
  cross_class_id = eval(args.cross_class_id)
  cls_wise_bbox_voting_thred = eval(args.cls_wise_bbox_voting_thred)


  if not osp.exists(args.dst_json_root):
    os.makedirs(args.dst_json_root)


  # 1. Do selection
  if select_res_from_diff_json:
    # pdb.set_trace()
    # print(select_method)

    assert len(src_json_roots) == len(select_method)

    res = select_res_from_jsons(src_json_roots, select_method)

  # pdb.set_trace()

  # 2. Do cross class suppression

  if cross_class_suppression:

    res = cross_class_suppress(res, cross_class_id)


  # 3. Do class-wise bbox voting
  if cls_wise_bbox_voting:

    res = class_wise_bbox_voting(res, cls_wise_bbox_voting_thred)


  # pdb.set_trace()

  # if args.cross_class_suppression:
  #   dump_to_json(sup_res, dst_json_root)
  # else:
  #   dump_to_json(select_res, dst_json_root)

  dump_to_json(res, dst_json_root)

  replace_json(dst_json_root)

  print("Totally suppress bboxes: {}".format(COUNTER))









