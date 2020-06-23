"""
  @lichenyang 2019.12.06
  The cervical cancer dataset for each sub class.
"""

import os
import cv2
import json
import pickle
import numpy as np
import os.path as osp

from .custom import CustomDataset
from .registry import DATASETS

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


@DATASETS.register_module
class CervicalCancerPosClsDataset(CustomDataset):

  CLASSES = (
    'ASC-H',    
    'ASC-US',   
    'HSIL',     
    'LSIL',     
    )

  def __init__(self, **args):

    args_cur = args.copy()
    image_size = args_cur.pop('image_size')
    self.WIDTH, self.HEIGHT = image_size

    super(CervicalCancerPosClsDataset, self).__init__(**args_cur)


  def load_annotations(self, ann_file):

    npz_names = os.listdir(ann_file)
    self.image_index = [name.split('.')[0] for name in npz_names]

    # img_infos = [self._load_annotation(index, ann_file) for index in self.image_index]

    img_infos = []

    for img_name in self.image_index:

      res = self._load_annotation(img_name, ann_file)

      if res is None: # The roi doesn't include four pos cls.
        continue

      img_infos.append(res)

    # pdb.set_trace()

    return img_infos




  def _load_annotation(self, index, ann_file, file_suffix='.npz'):
    
    filename = index + file_suffix
    height = self.HEIGHT
    width = self.WIDTH

    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
    gt_labels = np.zeros((0), dtype=np.int64)

    ann = dict(
      filename=filename,
      height=height,
      width=width,
      ann=dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        )
      )


    npz_dir = osp.join(ann_file, filename)
    data = np.load(npz_dir)
    label = data['label']

    keep_box_ids = np.where(label[:, 4] < 4)[0] # keep the boxes of four positive cls.

    if len(keep_box_ids) == 0:
      return None
    else:
      ann['keep_box_ids'] = keep_box_ids

    return ann

  def _class_to_ind(self, class_name):

    return CervicalCancerSixClsDataset.CLASSES.index(class_name) + 1



@DATASETS.register_module
class CervicalCancerCanClsDataset(CustomDataset):

  CLASSES = (
    'Candida',  
    )

  def __init__(self, **args):

    args_cur = args.copy()
    image_size = args_cur.pop('image_size')
    self.WIDTH, self.HEIGHT = image_size

    super(CervicalCancerCanClsDataset, self).__init__(**args_cur)


  def load_annotations(self, ann_file):

    npz_names = os.listdir(ann_file)
    self.image_index = [name.split('.')[0] for name in npz_names]

    # img_infos = [self._load_annotation(index, ann_file) for index in self.image_index]

    img_infos = []

    for img_name in self.image_index:

      res = self._load_annotation(img_name, ann_file)

      if res is None: # The roi doesn't include four pos cls.
        continue

      img_infos.append(res)

    # pdb.set_trace()

    return img_infos




  def _load_annotation(self, index, ann_file, file_suffix='.npz'):
    
    filename = index + file_suffix
    height = self.HEIGHT
    width = self.WIDTH

    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
    gt_labels = np.zeros((0), dtype=np.int64)

    ann = dict(
      filename=filename,
      height=height,
      width=width,
      ann=dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        )
      )


    npz_dir = osp.join(ann_file, filename)
    data = np.load(npz_dir)
    label = data['label']

    keep_box_ids = np.where(label[:, 4] == 4)[0] # keep the boxes of Candida.

    if len(keep_box_ids) == 0:
      return None
    else:
      ann['keep_box_ids'] = keep_box_ids
      ann['label_offset'] = 4

    return ann

  def _class_to_ind(self, class_name):

    return CervicalCancerSixClsDataset.CLASSES.index(class_name) + 1






@DATASETS.register_module
class CervicalCancerTriClsDataset(CustomDataset):

  CLASSES = (
    'Trichomonas',  
    )

  def __init__(self, **args):

    args_cur = args.copy()
    image_size = args_cur.pop('image_size')
    self.WIDTH, self.HEIGHT = image_size

    super(CervicalCancerTriClsDataset, self).__init__(**args_cur)


  def load_annotations(self, ann_file):

    npz_names = os.listdir(ann_file)
    self.image_index = [name.split('.')[0] for name in npz_names]

    # img_infos = [self._load_annotation(index, ann_file) for index in self.image_index]

    img_infos = []

    for img_name in self.image_index:

      res = self._load_annotation(img_name, ann_file)

      if res is None: # The roi doesn't include four pos cls.
        continue

      img_infos.append(res)

    # pdb.set_trace()

    return img_infos




  def _load_annotation(self, index, ann_file, file_suffix='.npz'):
    
    filename = index + file_suffix
    height = self.HEIGHT
    width = self.WIDTH

    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
    gt_labels = np.zeros((0), dtype=np.int64)

    ann = dict(
      filename=filename,
      height=height,
      width=width,
      ann=dict(
        bboxes=gt_bboxes,
        labels=gt_labels,
        )
      )


    npz_dir = osp.join(ann_file, filename)
    data = np.load(npz_dir)
    label = data['label']

    keep_box_ids = np.where(label[:, 4] == 5)[0] # keep the boxes of Trichomonas.

    if len(keep_box_ids) == 0:
      return None
    else:
      ann['keep_box_ids'] = keep_box_ids
      ann['label_offset'] = 5

    return ann

  def _class_to_ind(self, class_name):

    return CervicalCancerSixClsDataset.CLASSES.index(class_name) + 1





