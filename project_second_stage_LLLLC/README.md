



### 1 环境配置

本项目使用到的相关的环境配置如下：
* 本项目的代码是基于开源项目 [mmdetection](https://github.com/open-mmlab/mmdetection)
1. Red Hat 4.8.5-4
2. Python 3.6.4
3. PyTorch 1.1.0
4. CUDA 10.0 CUDNN 7.5.1



### 2 代码结构



|-- project_second_stage_LLLLC

​	|-- code

​		|-- data

​			|-- create_samples_npz_train.py (截取ROI并保存为npz的脚本)

​		|-- mmdetection

​		inference.sh (前向脚本)

​		train.sh (训练脚本)

​	|-- prediction_result (最终预测结果)

​	|-- user_data (保存中间数据)

​	README.md



### 3 程序运行         

训练：
1. `cd code`
2. `bash train.sh`

测试：
1. `cd code`
2. `bash inference.sh`



### 4 算法说明

详细参见 Readme 补充材料



### 5 其它注意事项

1. 本项目在PAI上的路径为：/home/admin/jupyter/Projects/project_second_stage_LLLLC
2. 训练和前向过程中使用到的模型放在：/home/admin/jupyter/Downloads/models/
3. `transforms.py`中的一些默认的npz已经事先生成好，生成的方法如下 (使用`./data/`下面的脚本)：
    1. `create_mixup_dict.py`: 生成"/home/admin/jupyter/Datasets/mix_up_dict.npz";
    2. `create_test_npz.py`: 生成"/home/admin/jupyter/Datasets/copy_pasting_dict.npz";
    3. `create_pseudo_label_npz.py`: 生成"/home/admin/jupyter/Datasets/pseudo_label/pseudo_label_dict_candida.npz";
    4. `create_src_kfb_img_size.py`: 生成"/home/admin/jupyter/Datasets/src_kfb_img_size.json";