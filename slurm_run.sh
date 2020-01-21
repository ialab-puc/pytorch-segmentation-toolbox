#!/bin/bash

#SBATCH --job-name=pspCityscapes       # Nombre del trabajo
#SBATCH --output=output/psp_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=output/err/psp_%j.err          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=8           # Numero de cores por tarea
#SBATCH --distribution=cyclic:cyclic # Distribuir las tareas de modo ciclico
#SBATCH --time=0-02:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=12000mb         # Memoria por proceso
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=afcadiz@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:4                 # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N)
#SBATCH --nodelist=hydra

CS_PATH=$1
MODEL=pspnet
LR=1e-2
WD=5e-4
BS=8
STEPS=40000
GPU_IDS=0

python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS} --restore-from ./dataset/MS_DeepLab_resnet_pretrained_init.pth --gpu ${GPU_IDS}
#python train.py --data-dir ${CS_PATH} --model ${MODEL} --random-mirror --random-scale --restore-from ./dataset/MS_DeepLab_resnet_pretrained_init.pth --gpu ${GPU_IDS} --learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --num-steps ${STEPS}
