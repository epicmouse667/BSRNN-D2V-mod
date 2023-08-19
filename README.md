# Wesep

> We aim to build a toolkit focusing on target speaker extraction, which is closely related to speech separation
## Features
- [ ] On the fly data simulation
- [ ] Support time- and frequency- domain models
- [ ] Low-latency streaming mode
- [ ] Clean and well-structured codes
- [ ] SOTA results

## Notes
Refer to Feishu https://yzsxeajuhm.feishu.cn/drive/folder/LMsLf8oT2lbE1rdBjgnceVz2nXc?from=from_copylink

## Data Pipe Design

<img src="resources/datapipe.png" width="800px">

### Shard Mode
In make_shard_list.py, we want to support variable number of speakers in the mixture

```python
# key: the name of the utterance
# spks: list of all speakers, obtained from the utt2spk
# wavs: all wavs including the mixture wav
key, spks, wavs = item
# we will rename each speaker and wav files
# tar file will encode a sample as 
# key: filename
# key.spk1: spk1 name
# key.spk2: spk2 name
# key.spk...
# key.wav: mixture wav data
# key_spk1.wav: spk1 wav data
# key_spk2.wav: spk2 wav data
# key_spk....wav
```

when we read from the shard file using the processor.tar_file_and_group
```python
# Obtain one sample and find the position by searching "."
# example[spk1], example[spk2], example[spk..]
# example[wav_mix], example[wav_spk1], example[wav_spk2], ...
# example[sample_rate]
# example[key]
```

## Model to Support
- [ ] Time domain model: Conv-tasnet
- [ ] Frequency domain model: Band-split RNN

## Integrating Speaker Information
- Pretrained speaker embedding
  - Wespeaker ResNet34 (pretrained on voxceleb)

- Joint Learning Speaker Embedding

- Embedding integration method
  - Concatenation
  - Conditional Layers (CLN, FiLM)

## Dataset to Support

For Target speaker extraction:
- [ ] WSJ0-2Mix
- [ ] Libri2Mix
