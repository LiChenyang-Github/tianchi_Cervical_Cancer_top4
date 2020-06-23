

DST_JSON_FOLDER=tri_fusion
TXT_ROOT_1=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_txt/tri_model_1
TXT_ROOT_2=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_txt/tri_model_2
TXT_ROOT_3=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_txt/tri_model_3
TXT_ROOT_4=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_txt/tri_model_4

JSON_ROOT=/home/admin/jupyter/Projects/project_second_stage_LLLLC/user_data/prediction/pred_json/${DST_JSON_FOLDER}




python ./gen_pred_json.py --txt_roots ${TXT_ROOT_1} ${TXT_ROOT_2} ${TXT_ROOT_3} ${TXT_ROOT_4} \
                          --json_root ${JSON_ROOT} \
                          --roi_fusion \
                          --model_fusion \
