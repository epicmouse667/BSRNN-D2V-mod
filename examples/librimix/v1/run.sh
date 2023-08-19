#!/bin/bash

# Copyright 2023 Shuai Wang (wangshuai@cuhk.edu.cn)

. ./path.sh || exit 1

stage=-1
stop_stage=1

data=data/clean
data_type="shard"  # shard/raw
gpus="[0,1]"
num_avg=5
checkpoint=
config=confs/convtasnet.yaml
exp_dir=exp/ConvTasnet/deep_encdec_FiLM_no_spktransform


. tools/parse_options.sh || exit 1


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
#  mkdir ${data}
  ./local/prepare_data.sh --data ${data} --stage 1 --stop-stage 2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in train-100 dev test; do
#  for dset in train-360; do
    python tools/make_shard_list_premix.py --num_utts_per_shard 1000 \
        --num_threads 16 \
        --prefix shards \
        --shuffle \
        ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
        ${data}/$dset/utt2trans
        ${data}/$dset/shards ${data}/$dset/shard.list
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#  rm -r $exp_dir
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wesep/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/train-460/${data_type}.list \
      --train_spk_embeds ${data}/train-460/embed.scp \
      --train_utt2spk ${data}/train-460/single.utt2spk \
      --val_data ${data}/dev/${data_type}.list \
      --val_spk_embeds ${data}/dev/embed.scp \
      --val_utt2spk ${data}/dev/single.utt2spk \
      --val_spk1_enroll ${data}/dev/spk1.enroll \
      --val_spk2_enroll ${data}/dev/spk2.enroll \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  python wesep/bin/infer.py --config $config \
      --gpus 1 \
      --exp_dir ${exp_dir} \
      --data_type "${data_type}" \
      --test_data ${data}/test/${data_type}.list \
      --test_spk_embeds ${data}/test/embed.scp \
      --test_spk1_enroll ${data}/test/spk1.enroll \
      --test_spk2_enroll ${data}/test/spk2.enroll \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wesep/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}
fi

#./run.sh --stage 4 --stop-stage 4 --config exp/BSRNN/train_clean_100/additive_no_spk_transform/config.yaml  --exp_dir exp/BSRNN/train_clean_100/additive_no_spk_transform/ --checkpoint exp/BSRNN/train_clean_100/additive_no_spk_transform/models/avg_model.pt