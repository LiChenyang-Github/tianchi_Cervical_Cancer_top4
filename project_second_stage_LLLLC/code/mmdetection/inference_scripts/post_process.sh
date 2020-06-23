#!/bin/bash

set -x




SRC_JSON_ROOT=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/model_combine
DST_JSON_ROOT=/home/admin/jupyter/Projects/project_second_stage_LLLLC/prediction_result/final_LLLLC



python ./post_process.py --pred_json_dir ${SRC_JSON_ROOT} \
                        --save_dir ${DST_JSON_ROOT} \