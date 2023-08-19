#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,4 ./run.sh --stage 3 --stop-stage 3 \
--config exp/BSRNN/train100/FiLM_no_spk_transform/config.yaml \
--exp_dir exp/BSRNN/train100/FiLM_no_spk_transform/ 
