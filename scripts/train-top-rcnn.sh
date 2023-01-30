#!/bin/bash
#SBATCH --job-name=train-top-rcnn
#SBATCH --account=project_2005430
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:50

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/datasets/alien-barley-2-train.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --num-gpus 4 \
      --config configs/Faster-RCNN/alien-barley/top_part.yaml
