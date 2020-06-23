


CONFIG_DIR="/home/admin/jupyter/Projects/project_second_stage_LLLLC/code/mmdetection/configs/Trichomonas/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1000x1000_mine.py"
CHECKPOINT="/home/admin/jupyter/Projects/project_second_stage_LLLLC/code/mmdetection/models/models_for_inference/Trichomonas/faster_rcnn_dconv_c3-c5_r50_fpn_1x_1000x1000_mine/epoch_60.pth"
SRC_IMAGE_ROOT="/home/admin/jupyter/Data/test/"
DST_FOLDER_NAME="tri_model_3"
DST_TXT_ROOT="/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_txt/${DST_FOLDER_NAME}"

WIN_SIZE=1000
WIN_STRIDE=500




START_ID=0
END_ID=350
python ./gen_pred_bbox.py \
    ${CONFIG_DIR} ${CHECKPOINT} \
    --src_image_root ${SRC_IMAGE_ROOT} --dst_txt_root ${DST_TXT_ROOT} \
    --pred_img_interval ${START_ID} ${END_ID} \
    --sliding_win_size ${WIN_SIZE} ${WIN_SIZE} \
    --sliding_win_stride ${WIN_STRIDE} ${WIN_STRIDE} \


