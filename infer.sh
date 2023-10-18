. ./path.sh
CUDA_VISIBLE_DEVICES=7    python wesep/bin/infer.py \
--checkpoint /workspace2/zixin/wesep/examples/librimix/v1/exp/BSRNN_D2V/train100/pipeline/models/model_73.pt \
--gpus 0 \
--config /workspace2/zixin/wesep/examples/librimix/v1/exp/BSRNN_D2V/train100/pipeline/config.yaml
