#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='RSSGL.RSSGL_Indian_Pines'

model_dir='./log/indianpines/RSSGL/poly'


python3 train_indianpines.py --config_path=${config_path} --model_dir=${model_dir} train.save_ckpt_interval_epoch 9999