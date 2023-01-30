#!/bin/bash
#SBATCH --job-name=ut2-test
#SBATCH --account=project_2005430
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4,nvme:50

module load pytorch
tar -xvf /scratch/project_2005430/ruastefa/datasets/alien-barley-0.0.1.tar -C $LOCAL_SCRATCH

srun python3 train_net.py \
      --num-gpus 4 \
      --config configs/FCOS/alien-barley/all_samples.yaml
