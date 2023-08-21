#!/bin/bash
CUDA_VISIBLE_DEVICES=0,2,4 ./run.sh --stage 3 --stop-stage 3 \
--config exp/BSRNN_D2V/train100/pretrained_extraction/config.yaml \
--exp_dir exp/BSRNN_D2V/train100/pretrained_extraction/ \
--gpus '[0,1,2]'