#!/bin/bash

python main_train.py \
--save_path exp/FastResNet_vox2_SD \
--n_class 5994 \
--nOut 256 \
--test_interval 1 \
--batch_size 128 \
--eval \

# python main_train.py \
# --save_path exp/ECAPATDNN512_vox2_SD \
# --n_class 5994 \
# --nOut 192 \
# --test_interval 1 \
# --batch_size 256 \

