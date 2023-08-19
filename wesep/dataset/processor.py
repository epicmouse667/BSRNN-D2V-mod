import io
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile

from wesep.dataset.FRA_RIR import FRA_RIR

# TODO:
# speaker_mix
# add_reverb (Yi Luo's online reverb generation toolkit)
# add_noise
# compute_spec


AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, mix_wav, spk1_wav, spk2_wav, ..., sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prev_prefix not in prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if 'spk' in postfix:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif 'trans' in postfix:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        if prefix[-5:-1] == '_spk':
                            example['wav' + prefix[-5:]] = waveform
                            prefix = prefix[:-5]
                        else:
                            example['wav_mix'] = waveform
                            example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix

        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           spk=spk,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wavs, spks}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wavs, spks}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        else:
            label = -1
        sample['label'] = label
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wavs, spks, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wavs, spks, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        sample_rate = sample['sample_rate']
        if sample_rate != resample_rate:
            all_keys = list(sample.keys())
            sample['sample_rate'] = resample_rate
            for key in all_keys:
                if 'wav' in key:
                    waveform = sample[key]
                    sample[key] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def sample_spk_embedding(data, spk_embeds):
    """ sample reference speaker embeddings for the target speaker
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            spk_embeds: dict which stores all potential embeddings for the speaker
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith('spk'):
                sample[key + "_embed"] = random.choice(spk_embeds[sample[key]])
        yield sample

def sample_fix_spk_embedding(data, spk2embed_dict, spk1_embed, spk2_embed):
    """ sample reference speaker embeddings for the target speaker
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            spk_embeds: dict which stores all potential embeddings for the speaker
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith('spk'):
                if key=='spk1':
                    sample[key + "_embed"] = spk2embed_dict[spk1_embed[sample['key']]]
                else:
                    sample[key + "_embed"] = spk2embed_dict[spk2_embed[sample['key']]]
        yield sample

def get_random_chunk(data_list, chunk_len):
    """ Get random chunk

        Args:
            data_list: [torch.Tensor: 1XT] (random len)
            chunk_len: chunk length

        Returns:
            [torch.Tensor] (exactly chunk_len)
    """
    # Assert all entries in the list share the same length
    assert False not in [len(i) == len(data_list[0]) for i in data_list]
    data_list = [data[0] for data in data_list]

    data_len = len(data_list[0])

    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        for i in range(len(data_list)):
            data_list[i] = data_list[i][chunk_start:chunk_start + chunk_len]
            # re-clone the data to avoid memory leakage
            if type(data_list[i]) == torch.Tensor:
                data_list[i] = data_list[i].clone()
            else:  # np.array
                data_list[i] = data_list[i].copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        for i in range(len(data_list)):
            if type(data_list[i]) == torch.Tensor:
                data_list[i] = data_list[i].repeat(repeat_factor)
            else:  # np.array
                data_list[i] = np.tile(data_list[i], repeat_factor)
            data_list[i] = data_list[i][:chunk_len]
    data_list = [data.unsqueeze(0) for data in data_list]
    return data_list


def filter(data,
           min_num_frames=100,
           max_num_frames=800,
           frame_shift=10,
           data_type='shard/raw'
           ):
    """ Filter the utterance with very short duration and random chunk the
        utterance with very long duration.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            min_num_frames: minimum number of frames of acoustic features
            max_num_frames: maximum number of frames of acoustic features
            frame_shift: the frame shift of the acoustic features (ms)
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            if len(feat) < min_num_frames:
                continue
            elif len(feat) > max_num_frames:
                feat = get_random_chunk(feat, max_num_frames)
            sample['feat'] = feat
        else:
            assert 'sample_rate' in sample
            assert 'wav' in sample
            sample_rate = sample['sample_rate']
            wav = sample['wav'][0]

            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)
            max_len = int(frame_shift / 1000 * max_num_frames * sample_rate)

            if len(wav) < min_len:
                continue
            elif len(wav) > max_len:
                wav = get_random_chunk(wav, max_len)
            sample['wav'] = wav.unsqueeze(0)

        yield sample


def random_chunk(data, chunk_len):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        wav_keys = [key for key in list(sample.keys()) if 'wav' in key]
        wav_data_list = [sample[key] for key in wav_keys]
        wav_data_list = get_random_chunk(wav_data_list, chunk_len)
        sample.update(zip(wav_keys, wav_data_list))
        # sample['wav'] = wav.unsqueeze(0)
        yield sample


def fix_chunk(data, chunk_len):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        all_keys = list(sample.keys())
        for key in all_keys:
            if key.startswith('wav'):
                sample[key] = sample[key][:, :chunk_len]
        yield sample


def add_reverb(data):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        reverb_mix = np.zeros(len(sample['wav_mix'][0]))
        for key in list(sample.keys()):

            if 'wav' in key:
                if 'spk' in key:
                    audio = sample[key].numpy()
                    audio_len = len(audio[0])

                    rir, _ = FRA_RIR(sr=audio_len)

                    rir_audio = signal.fftconvolve(audio, rir)

                    max_scale = np.max(np.abs(rir_audio))
                    out_audio = rir_audio / max_scale * 0.9

                    def pad_sig(x):
                        if len(x[0]) < audio_len:
                            zeros = np.zeros(audio_len - len(x[0]))
                            print(x)
                            return np.concatenate([x[0], zeros])
                        else:
                            return x[0][:audio_len]

                    out_audio = pad_sig(out_audio)

                    sample[key + '_reverb'] = torch.from_numpy(out_audio).unsqueeze(0)
                    reverb_mix += out_audio

                    max_scale = np.max(np.abs(reverb_mix))
                    reverb_out_audio = reverb_mix / max_scale * 0.9

                    reverb_mix = reverb_out_audio
                    # torchaudio.save('/workspace2/yixuan/wesep/examples/librimix/v1/exp_reverb/data_reverb/sample{}.wav'.format(key), sample[key+ '_reverb'], 16000)

        sample['wav_mix'] = torch.from_numpy(reverb_mix).unsqueeze(0)
        # print('wav_mix', sample['wav_mix'])
        # sf.write('/workspace2/yixuan/wesep/examples/librimix/v1/exp_reverb/data_reverb/sample{}.wav'.format(key+'_reverb'), sample['wav_mix'][0], 16000)

        yield sample