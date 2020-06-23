"""
  @lichenyang 2019.11.29
  The cervical cancer dataset for second stage.
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
class CervicalCancerSixClsDataset(CustomDataset):
  """
    Note: 
      Because the .npz file is to large. We do not load them here.
  """

  CLASSES = (
    'ASC-H',    
    'ASC-US',   
    'HSIL',     
    'LSIL',     
    'Candida',  
    'Trichomonas',  
    )

  def __init__(self, **args):

    args_cur = args.copy()
    image_size = args_cur.pop('image_size')
    self.WIDTH, self.HEIGHT = image_size

    if 'sampling_files' in args_cur:
      self.sampling_files = args_cur.pop('sampling_files')
      self.sampling_p = args_cur.pop('sampling_p')
      self.sampling_files_root = args_cur.pop('sampling_files_root')

      self.sampling_npz_names = self._get_sampling_npz_names()
    else:
      self.sampling_npz_names = None


    super(CervicalCancerSixClsDataset, self).__init__(**args_cur)


  def load_annotations(self, ann_file):

    # self.HEIGHT = 800 ### Note: It is just be used in custom.py
    # self.WIDTH = 800

    npz_names = os.listdir(ann_file)
    self.image_index = [name.split('.')[0] for name in npz_names]

    img_infos = [self._load_annotation(index, ann_file) for index in self.image_index]

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

    return ann

  def _class_to_ind(self, class_name):

    return CervicalCancerSixClsDataset.CLASSES.index(class_name) + 1

  def _get_sampling_npz_names(self):
    res = []

    for i in range(len(self.sampling_files)):
      json_dir = osp.join(self.sampling_files_root, self.sampling_files[i])
      with open(json_dir, 'r') as f:
        json_info = json.load(f)
      res.append(json_info)

    return res


  def prepare_train_img(self, idx):
      img_info = self.img_infos[idx]
      ann_info = self.get_ann_info(idx)

      ### Add sampling code (random dropping).
      if self.sampling_npz_names is not None:
        for i in range(len(self.sampling_npz_names)):
          if img_info['filename'] in self.sampling_npz_names[i]:
            p = np.random.rand()
            if p < self.sampling_p[i]:
              return None

      results = dict(img_info=img_info, ann_info=ann_info)
      if self.proposals is not None:
          results['proposals'] = self.proposals[idx]
      self.pre_pipeline(results)
      return self.pipeline(results)

