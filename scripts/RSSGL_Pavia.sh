#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='RSSGL.RSSGL_Pavia'

model_dir='./log/pavia/RSSGL/poly'


python3 train_pavia.py --config_path=${config_path} --model_dir=${model_dir} train.save_ckpt_interval_epoch 9999
