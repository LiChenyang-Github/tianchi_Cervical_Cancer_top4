"""
  @lichenyang 2019.11.16
  The cervical cancer dataset.
"""

import os
import cv2
import json
import pickle
import numpy as np
import os.path as osp

from .custom import CustomDataset
from .registry import DATASETS

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


@DATASETS.register_module
class CervicalCancerDataset(CustomDataset):

  CLASSES = ('pos', )


  def load_annotations(self, ann_file):

    self.HEIGHT = 1000
    self.WIDTH = 1200
    cache_path = "./data/cache"
    cache_file = osp.join(cache_path, self.__class__.__name__ + '_gt_roidb.pkl')

    # Cache file already exist.
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        img_infos = pickle.load(fid)
        print('{} gt roidb loaded from {}'.format(self.__class__.__name__, cache_file))
        return img_infos

    json_names = os.listdir(ann_file)
    self.image_index = [name.split('.')[0] for name in json_names]

    img_infos = [self._load_annotation(index, ann_file) for index in self.image_index]

    if not osp.exists(cache_path):
      os.makedirs(cache_path)
    with open(cache_file, 'wb') as fid:
      pickle.dump(img_infos, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return img_infos




  def _load_annotation(self, index, ann_file, img_suffix='.jpg'):
    
    filename = index + img_suffix
    height = self.HEIGHT
    width = self.WIDTH

    json_dir = osp.join(ann_file, index + '.json')
    with open(json_dir, 'r') as f:
        labels = json.load(f)
        objs = labels['pos_list']
    num_objs = len(objs)

    gt_bboxes = np.zeros((num_objs, 4), dtype=np.float32)
    gt_labels = np.zeros((num_objs), dtype=np.int64)
    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):

        x1 = obj['x']
        y1 = obj['y']
        x2 = x1 + obj['w'] - 1
        y2 = y1 + obj['h'] - 1

        cls = self._class_to_ind(obj['class'])
        gt_bboxes[ix, :] = [x1, y1, x2, y2]
        gt_labels[ix] = cls


    ann = dict(
      filename=filename,
      height=height,
      width=width,
      ann=dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        bboxes_ignore=gt_bboxes_ignore,
        )
      )

    return ann

  def _class_to_ind(self, class_name):

    return CervicalCancerDataset.CLASSES.index(class_name) + 1




