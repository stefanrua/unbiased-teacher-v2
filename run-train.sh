#!/bin/bash
#SBATCH --job-name=alien-barley-train
#SBATCH --account=project_2005430
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:20

tar -xf /scratch/project_2005430/ruastefa/datasets/alien-barley-labeled.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --num-gpus 4 \
      --config configs/Faster-RCNN/alien-barley/faster_rcnn_R_50_FPN_ut2_sup10_run0.yaml
