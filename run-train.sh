#!/bin/bash
python3 train_net.py \
      --num-gpus 1 \
      --config configs/Faster-RCNN/alien-barley/faster_rcnn_R_50_FPN_ut2_sup10_run0.yaml
