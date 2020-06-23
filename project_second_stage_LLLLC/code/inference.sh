#!/bin/bash

set -x




##################################################
# step 1: generate predicted results into txt.
##################################################

cd mmdetection


inference_func_1(){
    {
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/pos_model_1.sh
    }&
    
    {
        sleep 10    # sleep to avoid GPU OOM.
        CUDA_VISIBLE_DEVICES=1 bash ./inference_scripts/pos_model_2.sh
    }&

    {
        sleep 20
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/pos_model_3.sh
    }&

    {
        sleep 30
        CUDA_VISIBLE_DEVICES=1 bash ./inference_scripts/pos_model_4.sh
    }&

    {
        sleep 40
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/tri_model_1.sh
    }&

    {
        sleep 50
        CUDA_VISIBLE_DEVICES=1 bash ./inference_scripts/tri_model_3.sh
    }&

}


inference_func_2(){
    {
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/tri_model_2.sh
    }&
    
    {
        sleep 10
        CUDA_VISIBLE_DEVICES=1 bash ./inference_scripts/tri_model_4.sh
    }&

    {
        sleep 20
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/can_model_1.sh
    }&

    {
        sleep 30
        CUDA_VISIBLE_DEVICES=1 bash ./inference_scripts/can_model_2.sh
    }&

    {
        sleep 40
        CUDA_VISIBLE_DEVICES=0 bash ./inference_scripts/can_model_3.sh
    }&

}


inference_func_1
wait
inference_func_2
wait



####################################################
# step 2: overlap area bbox voting and model fusion
# txt to json.
####################################################

(bash ./inference_scripts/pos_model_fusion.sh & \

bash ./inference_scripts/tri_model_fusion.sh & \

bash ./inference_scripts/can_model_fusion.sh)

wait

####################################################
# step 3: post processing.
####################################################

bash ./inference_scripts/combine_json.sh

bash ./inference_scripts/post_process.sh




