#!/bin/bash
#SBATCH --job-name=test-bottom-rcnn
#SBATCH --account=project_2005430
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:50

module load pytorch
tar -xf /scratch/project_2005430/ruastefa/datasets/alien-barley-2-test.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --num-gpus 4 \
      --config configs/Faster-RCNN/alien-barley/bottom_part.yaml \
      --eval-only \
      MODEL.WEIGHTS output/model_0000999.pth
