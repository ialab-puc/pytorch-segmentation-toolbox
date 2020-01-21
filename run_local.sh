#!/bin/bash
uname -a
#date
#env
date

CS_PATH=$1
MODEL=pspnet
LR=1e-2
WD=5e-4
BS=8
STEPS=40000
GPU_IDS=0

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --restore-from ./dataset/MS_DeepLab_resnet_pretrained_init.pth
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/MS_DeepLab_resnet_pretrained_init.pth --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS}
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch --nproc_per_node=4 evaluate.py --data-dir ${CS_PATH} --model ${MODEL} --input-size ${INPUT_SIZE} --batch-size 4 --restore-from snapshots/CS_scenes_${STEPS}.pth --gpu 0
