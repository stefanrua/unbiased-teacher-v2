#!/bin/bash
#SBATCH --job-name=alien-barley-test
#SBATCH --account=project_2005430
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2,nvme:20

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/datasets/alien-barley-1.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --eval-only \
      --num-gpus 2 \
      --config configs/Faster-RCNN/alien-barley/all_samples.yaml \
      MODEL.WEIGHTS outputs/all-train/model_0107999.pth
