#!/bin/bash

set -x


###########################
# step 1: create roi npz.
###########################

# python ./data/create_samples_npz_train.py


###########################
# step 2: train the models.
###########################

cd mmdetection

train_func_1(){
    {
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/PositiveCell/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1600x1600_multi_scale_mine.py & \
        CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/PositiveCell/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1200x1200_mine.py
    }&

    {
        sleep 20
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/PositiveCell/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1200x1200_multi_scale_mine.py & \
        CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/PositiveCell/cascade_rcnn_r50_fpn_1x_1200x1200_multi_scale_mine.py
    }

}

train_func_2(){
    {
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/Trichomonas/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1600x1600_mine.py & \
        CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/Trichomonas/cascade_rcnn_r50_fpn_1x_1600x1600_mine.py
    }&

    {
        sleep 20
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/Trichomonas/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1000x1000_mine.py & \
        CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/Trichomonas/cascade_rcnn_r50_fpn_1x_1000x1000_bifpn_mine.py
    }


}

train_func_3(){
    {
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/Candida/faster_rcnn_dconv_c3-c5_r50_fpn_1x_3000-2000_mine.py & \
        CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/Candida/faster_rcnn_dconv_c3-c5_r50_fpn_1x_2000x2000_mine.py
    }

    {
        CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/Candida/faster_rcnn_dconv_c3-c5_r50_fpn_1x_4000-2000_mine.py
    }
}


train_func_1
wait
train_func_2
wait
train_func_3

