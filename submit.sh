#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3,4 ./run.sh --stage 3 --stop-stage 3 \
--config exp/BSRNN_D2V/train100/pipeline/config.yaml \
--exp_dir exp/BSRNN_D2V/train100/pipeline \
--gpus '[0,1,2]' \
--checkpoint exp/BSRNN_D2V/train100/pipeline/models/model_70.pt
