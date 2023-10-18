#!/bin/bash
# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)

stage=-1
stop_stage=-1

mix_data_path='/workspace2/zixin/Datasets/Libri2Mix/wav16k/max'
librispeech_path=/workspace2/zixin/Datasets/LibriSpeech

data=
. tools/parse_options.sh || exit 1

data=data
datatype=clean

data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the datasets"
  for dataset in train-100 dev test; do
#  for dataset in train-360; do
    echo "Preparing files for" $dataset
    # Prepare the meta data for the mixed data
    dataset_path=$mix_data_path/$dataset/mix_${datatype}
    mkdir -p "${data}"/$datatype/${dataset}
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print $NF}' |
      awk -v path="${dataset_path}" '{print $1 , path "/" $1 , path "/../s1/" $1 , path "/../s2/" $1}' |
      sed 's#.wav##' >"${data}"/$datatype/${dataset}/wav.scp
    awk '{print $1}' "${data}"/$datatype/${dataset}/wav.scp |
      awk -F[_-] '{print $0, $1,$4}' >"${data}"/$datatype/${dataset}/utt2spk
    
    if [[ $dataset == train* ]]; 
    then
      split='train-clean-100'
    elif [[ $dataset == dev* ]]; 
    then
      split='dev-clean'
    else
      split='test-clean'
    fi
    awk '{print $1}' "${data}"/$datatype/${dataset}/wav.scp |
      awk -F[_-] -v librispeech_path="$librispeech_path" -v   splitt="$split" '{
        print $0,
      librispeech_path"/"splitt"/"$1"/"$2"/"$1"-"$2".trans.txt",$3,
      librispeech_path"/"splitt"/"$4"/"$5"/"$4"-"$5".trans.txt",$6
      }' >"${data}"/$datatype/${dataset}/utt2trans.txt

    # Prepare the meta data for single speakers
    dataset_path=$mix_data_path/$dataset/s1
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s1/" $NF, $0}' > "${data}"/$datatype/${dataset}/single.wav.scp
    awk '{print $1}' "${data}"/$datatype/${dataset}/single.wav.scp | grep 's1' |
      awk -F[-_/] '{print $0, $2}' > "${data}"/$datatype/${dataset}/single.utt2spk

    dataset_path=$mix_data_path/$dataset/s2
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s2/" $NF, $0}' >> "${data}"/$datatype/${dataset}/single.wav.scp

    awk '{print $1}' "${data}"/$datatype/${dataset}/single.wav.scp | grep 's2' |
      awk -F[-_/] '{print $0, $5}' >> "${data}"/$datatype/${dataset}/single.utt2spk
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Prepare the speaker embeddings using wespeaker pretrained models"
  for dataset in dev test train-100; do
    # for dataset in train-360; do
    mkdir -p ${data}/$datatype/${dataset}
    echo "Preparing files for" $dataset
    python tools/extract_embed_premix.py --wav_scp "${data}"/$datatype/${dataset}/single.wav.scp \
      --onnx_path data/voxceleb_resnet34_LM.onnx \
      --out_path ${data}/$datatype/${dataset}
  done
fi


#for file in shard.list single.utt2spk single.wav.scp utt2spk wav.scp; do
#  cat train-100/$file train-360/$file > train-460/$file;
#done