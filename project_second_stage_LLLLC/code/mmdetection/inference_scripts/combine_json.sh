#!/bin/bash

set -x




DST_JSON_FOLDER=model_combine
SRC_JSON_ROOT_1=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/pos_fusion
SRC_JSON_ROOT_2=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/can_fusion
SRC_JSON_ROOT_3=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/tri_fusion

SELECT_METHOD="[(1,2,3,4),(5,),(6,)]"  # 1-based

# CROSS_CLASS_ID="(1,2,3,4)"  # 1-based
# CLS_WISE_BBOX_VOTING_THRED="(0.2,0.2,0.2,0.2,0.2,0.05)"




DST_JSON_ROOT=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/${DST_JSON_FOLDER}



python ./post_process_json.py --src_json_roots ${SRC_JSON_ROOT_1} ${SRC_JSON_ROOT_2} ${SRC_JSON_ROOT_3} \
                            --dst_json_root ${DST_JSON_ROOT} \
                            --select_res_from_diff_json \
                            --select_method ${SELECT_METHOD} \
                            # --cls_wise_bbox_voting \
                            # --cls_wise_bbox_voting_thred ${CLS_WISE_BBOX_VOTING_THRED} \
                            # --cross_class_suppression \
                            # --cross_class_id ${CROSS_CLASS_ID}


