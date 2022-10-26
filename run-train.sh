#!/bin/bash
#SBATCH --job-name=alien-barley-train
#SBATCH --account=project_2005430
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:50

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/datasets/alien-barley.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --resume \
      --num-gpus 4 \
      --config configs/Faster-RCNN/alien-barley/all_samples.yaml
