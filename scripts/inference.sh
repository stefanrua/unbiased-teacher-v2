#!/bin/bash
#SBATCH --job-name=barley-inference
#SBATCH --account=project_2005430
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:20

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/datasets/temp.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/Faster-RCNN/alien-barley/all_samples.yaml \
      MODEL.WEIGHTS output/model_0107999.pth \
      DATASETS.TEST "('inference',)"
