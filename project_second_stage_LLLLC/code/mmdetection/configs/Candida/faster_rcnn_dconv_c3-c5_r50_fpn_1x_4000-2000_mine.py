# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False, deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            gpu_assign_thr=10),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            gpu_assign_thr=10),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
)
# dataset settings
dataset_type = 'CervicalCancerCanClsDataset'  ####
data_root = 'data/npz_20191130_update'    ###



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)


crop_size = (4000, 4000)
image_size = (2000, 2000)   ###

import cv2
train_pipeline = [
    dict(type='LoadImageLabelFromNpz'),
    dict(type='GtBoxBasedCrop', crop_size=crop_size),
    dict(type='RandomShiftGtBBox', shift_rate=0.2),
    dict(type='ReplaceBackgroundCandida', crop_size=crop_size, drop_rate=0.8),
    dict(type='Resize', img_scale=image_size, keep_ratio=True),
    dict(type='Bboxes_Jitter'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomVerticalFlip', flip_ratio=0.5),
    dict(type='AlbuMine',
        transforms=[
            dict(
            type='ShiftScaleRotate',
            shift_limit=0.,
            scale_limit=0.,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5)
        ],
        bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    min_area=0.,
                    min_visibility=0.,
                    label_fields=['gt_labels'])),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        # img_scale=[(4000, 1550), (4000, 1600), (4000, 1650)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root,  ###
        img_prefix=data_root,  ###
        pipeline=train_pipeline,
        image_size=image_size),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline))


# optimizer
optimizer = dict(type='SGD', lr=0.02*(data['imgs_per_gpu']/16), momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


lr_config = dict(
    policy='cosine',
    target_lr=0.00005,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3)    ###

checkpoint_config = dict(interval=5)    ###
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 30
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './work_dirs/Candida/faster_rcnn_dconv_c3-c5_r50_fpn_1x_2nd_stage_update_4000-2000_bs1_multi_scale8-16-32-64_dr0.8'
load_from = './models/faster_r50_1600x1600_epoch_60.pth'
resume_from = None

workflow = [('train', 1)]

SEED = 1333