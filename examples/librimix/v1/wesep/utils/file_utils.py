import math

import kaldiio
import numpy as np


def read_lists(list_file):
    """ list_file: only 1 column
    """
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_vec_scp_file(scp_file):
    """
    Read the pre-extracted kaldi-format speaker embeddings.
    :param scp_file: path to xvector.scp
    :return: dict {wav_name: embedding}
    """
    samples_dict = {}
    for key, vec in kaldiio.load_scp_sequential(scp_file):
        samples_dict[key] = vec
    return samples_dict


def norm_embeddings(embeddings, kaldi_style=True):
    """
    Norm embeddings to unit length
    :param embeddings: input embeddings
    :param kaldi_style: if true, the norm should be embedding dimension
    :return:
    """
    scale = math.sqrt(embeddings.shape[-1]) if kaldi_style else 1.
    if len(embeddings.shape) == 2:
        return (scale * embeddings.transpose() /
                np.linalg.norm(embeddings, axis=1)).transpose()
    elif len(embeddings.shape) == 1:
        return scale * embeddings / np.linalg.norm(embeddings)


def read_label_file(label_file):
    """
    Read the utt2spk file
    :param label_file: the path to utt2spk
    :return: dict {wav_name: spk_id}
    """
    labels_dict = {}
    with open(label_file, 'r') as fin:
        for line in fin:
            tokens = line.strip().split()
            labels_dict[tokens[0]] = tokens[1]
    return labels_dict


def load_speaker_embeddings(scp_file, utt2spk_file):
    """
    :param scp_file:
    :param utt2spk_file:
    :return: {spk1: [emb1, emb2 ...], spk2: [emb1, emb2...]}
    """
    samples_dict = read_vec_scp_file(scp_file)
    labels_dict = read_label_file(utt2spk_file)
    spk2embeds = {}
    for key, vec in samples_dict.items():
        label = labels_dict[key]
        if label in spk2embeds.keys():
            spk2embeds[label].append(vec)
        else:
            spk2embeds[label] = [vec]
    return spk2embeds


