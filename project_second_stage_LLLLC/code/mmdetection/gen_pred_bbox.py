"""
    @lichenyang 2019.12.01
    Refer to demo/webcam_demo.py

    Generate the predicted bboxes for all test imgs, write the result into the corresponding txt.

    Step:
    1. Load roi from kfb img.
    2. Crop wins from roi img.
    3. Do inference for each win.
    4. Dump to txt.
"""

import argparse

import os
import os.path as osp
import cv2
import pdb
import json
import torch
import numpy as np
import kfbReader as kr

from tqdm import tqdm
from glob import glob


from mmdet.apis import inference_detector, init_detector, show_result

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--score_thr', type=float, default=0.05, help='bbox score threshold')


    parser.add_argument('--src_image_root', dest='src_image_root',
                      help='directory to load images for demo',
                      default="images")
    parser.add_argument('--dst_txt_root', dest='dst_txt_root',
                      help='directory to write result txt for demo',
                      default="images")
    parser.add_argument('--pred_img_interval', dest='pred_img_interval', \
                      nargs='+', type=int, default=0, help='pred img id interval')  
    parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
    parser.add_argument('--vis_image_dir', dest='vis_image_dir',
                      help='directory to write vis images for demo',
                      default="/home/admin/jupyter/Projects/mmdetection/output/visualization/")
    parser.add_argument('--update_dir', dest='update_dir',
                      help='directory to the update json file',
                      default="/home/admin/jupyter/Projects/mmdetection/data/update/update_test")

    parser.add_argument('--sliding_win_size', dest='sliding_win_size', \
                      nargs='+', type=int, default=[800, 800], help='The size of sliding windows.')  
    parser.add_argument('--sliding_win_stride', dest='sliding_win_stride', \
                      nargs='+', type=int, default=[400, 400], help='The stride of sliding windows.')  

    parser.add_argument('--label_offset', dest='label_offset', \
                      type=int, default=0, help='The label offset for the sub cls model.')  

    args = parser.parse_args()
    return args


def write_txt(txt_dir, res_dict):
    """
        Write the res in res_dict into txt_dir.
    """

    with open(txt_dir, 'w') as f:
        for k in res_dict.keys():
            for i in range(res_dict[k].shape[0]):

                x1 = str(res_dict[k][i][0].item())
                y1 = str(res_dict[k][i][1].item())
                x2 = str(res_dict[k][i][2].item())
                y2 = str(res_dict[k][i][3].item())
                score = str(res_dict[k][i][4].item())
                cls_id = k

                # print(x1, y1, x2, y2, score, cls_id)

                line = ' '.join([x1, y1, x2, y2, cls_id, score]) + '\n'

                f.write(line)



def gen_sliding_windows(org_img_dir, update_dir, update_json_names, \
                        win_size=(800, 800), stride=(400, 400)):
    """
        Generate sliding wins.
        - outputs:
            res: dict
                key: roi_name
                value: list(tuple): Each tuple is (img, (ltx, lty)), 'lt' means the left top coord in the src img.
    """                 

    res = {}
    src_img_name = osp.basename(org_img_dir).split('.')[0]

    # use update json
    if '{}.json'.format(src_img_name) in update_json_names:
        json_dir = osp.join(update_dir, '{}.json'.format(src_img_name))
        # print("Use update file: {}.".format(json_dir))
    else:
        json_dir = osp.join(osp.dirname(org_img_dir), '{}.json'.format(src_img_name))

    # print("Before")
    roi_tuples = get_roi_from_src_img(org_img_dir, json_dir)
    # print("After")


    # pdb.set_trace()

    for i, roi_tuple in enumerate(roi_tuples):
        roi_name = '{}_{}'.format(src_img_name, str(i+1))

        win_tuples = get_win_from_roi(roi_tuple, win_size=win_size, stride=stride)
        roi_h, roi_w, _ = roi_tuple[0].shape

        # res[roi_name] = (win_tuples, (roi_w, roi_h))
        res[roi_name] = win_tuples

        # pdb.set_trace()

    return res


def load_json(json_path):
    with open(json_path, "r") as f:
        js = json.load(f)
    return js

def get_roi_from_src_img(cfb_dir, json_dir, scale=20):
    """
        Get roi info from a src img.
        - outputs:
            res: list(tuple). Each correspond to a roi, (roi_img, (ltx, lty))
    """

    res = []
    json_infos = load_json(json_dir)

    kfb_reader = kr.reader()
    kr.reader.ReadInfo(kfb_reader, kfbPath=cfb_dir, scale=scale, readAll=False)

    for json_info in json_infos:
        if json_info['class'] == 'roi':
            # print("Before: ", json_info)
            roi_image = kfb_reader.ReadRoi(json_info["x"], json_info["y"], json_info["w"], json_info["h"], scale)
            # print("After: ", json_info)

            res.append((roi_image.copy(), (json_info["x"], json_info["y"])))

    return res


def get_win_from_roi(roi_tuple, win_size, stride):
    """
        Get the sliding wins in a roi.
        - inputs:
            roi_tuple: (roi_img, (ltx, lty))
            win_size: (x, y)
            stride: (stride_x, stride_y)
        - outputs:
            res: list(tuple). Each tuple is for a sliding win. (win_img, (ltx, lty))
    """

    res = []

    roi_img = roi_tuple[0]
    roi_ltx, roi_lty = roi_tuple[1]
    roi_h, roi_w, _ = roi_img.shape

    target_w, target_h = win_size
    stride_w, stride_h = stride

    # print(target_w, roi_w, target_h, roi_h)
    # assert target_w<=roi_w and target_h<=roi_h

    if target_w > roi_w or target_h > roi_h:
        pad_w = 0
        pad_h = 0
        if target_w > roi_w:
            pad_w = target_w - roi_w
            roi_w = target_w
        if target_h > roi_h:
            pad_h = target_h - roi_h
            roi_h = target_h
        roi_img = cv2.copyMakeBorder(roi_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, 0)

    new_w = target_w + np.ceil((roi_w - target_w) / stride_w) * stride_w
    new_h = target_h + np.ceil((roi_h - target_h) / stride_h) * stride_h

    stride_w_num = int((new_w - target_w) // stride_w + 1)
    stride_h_num = int((new_h - target_h) // stride_h + 1)

    # padding the img at right bottom
    # print("Before copyMakeBorder")
    # print(new_h, roi_h, new_w, roi_w, roi_img.shape, type(roi_img[0,0,0]))
    roi_img_pad = cv2.copyMakeBorder(roi_img,0,int(new_h-roi_h),0,int(new_w-roi_w),cv2.BORDER_CONSTANT,0)
    # print("After copyMakeBorder")


    # pdb.set_trace()

    for i in range(stride_w_num):
        for j in range(stride_h_num):
            win_topleft_x, win_topleft_y = i*stride_w, j*stride_h
            win_img = roi_img_pad[win_topleft_y: win_topleft_y+target_h, \
                            win_topleft_x: win_topleft_x+target_w, \
                            :]
            res.append((win_img.copy(), (roi_ltx+win_topleft_x, roi_lty+win_topleft_y)))

    return res







def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    id_interval = args.pred_img_interval


    if args.vis:
        if not osp.exists(args.vis_image_dir):
            os.makedirs(args.vis_image_dir)
    if not osp.exists(args.dst_txt_root):
        os.makedirs(args.dst_txt_root)
    if args.update_dir is not None:
        update_json_names = os.listdir(args.update_dir)
        print("Update json files: {}".format(update_json_names))

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    # pdb.set_trace()

    org_img_dirs = glob(osp.join(args.src_image_root, '*.kfb'))
    org_img_dirs = sorted(org_img_dirs, key=lambda k: int(k.split('/')[-1].split('.')[0]))

    # pdb.set_trace()


    print("There are {} original images.".format(len(org_img_dirs)))

    for i in tqdm(range(id_interval[0], id_interval[1])):   # Loop for ID interval
        org_img_dir = org_img_dirs[i]
        org_img_name = osp.basename(org_img_dir).split('.')[0]

        dst_txt_dir = osp.join(args.dst_txt_root, org_img_name)

        # generate sliding windows for each roi in current src img.
        # print("Before")
        src_sliding_dict = gen_sliding_windows(org_img_dir, args.update_dir, update_json_names, \
                                args.sliding_win_size, args.sliding_win_stride)
        # print("After")

        for roi_name in src_sliding_dict.keys():    # Loop for rois in a src img

            for win_tuple in src_sliding_dict[roi_name]:  # Loop for sliding wins in a roi
                win_img = win_tuple[0]
                ltx = win_tuple[1][0]
                lty = win_tuple[1][1]

                dst_txt_name = roi_name + '_{}_{}.txt'.format(ltx, lty)
                dst_sliding_win_txt_dir = osp.join(dst_txt_dir, roi_name, dst_txt_name)

                if not osp.exists(osp.dirname(dst_sliding_win_txt_dir)):
                    os.makedirs(osp.dirname(dst_sliding_win_txt_dir))

                result = inference_detector(model, win_img)

                res_dict = {}
                for j in xrange(0, len(model.CLASSES)):

                    cur_res = result[j]
                    inds = np.nonzero(cur_res[:,-1]>args.score_thr)[0].reshape(-1)

                    if inds.shape[0] > 0:
                        if args.label_offset > 0:
                            res_dict[str(j + 1 + args.label_offset)] = cur_res[inds]  # "+1" to let cls_id start from 1.
                        else:
                            res_dict[str(j + 1)] = cur_res[inds]  # "+1" to let cls_id start from 1.
                    # pdb.set_trace()

                if args.vis:
                    result_path = os.path.join(args.vis_image_dir, "{}_det.jpg".format(dst_txt_name.split('.')[0]))
                    show_result(win_img, \
                        result, model.CLASSES, \
                        score_thr=args.score_thr, show=False, out_file=result_path)

                write_txt(dst_sliding_win_txt_dir, res_dict)





if __name__ == '__main__':
    main()
