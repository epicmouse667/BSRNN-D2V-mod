# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shuai Wang (wsstriving@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wesep.dataset.processor as processor
from wesep.utils.file_utils import read_lists


def tse_collate_fn_2spk(batch):
    new_batch = {}

    wav_mix = []
    wav_targets = []
    spk_embeds = []
    for s in batch:
        wav_mix.append(s['wav_mix'])
        wav_targets.append(s['wav_spk1'])
        spk_embeds.append(torch.from_numpy(s['spk1_embed'].copy()))

        wav_mix.append(s['wav_mix'])
        wav_targets.append(s['wav_spk2'])
        spk_embeds.append(torch.from_numpy(s['spk2_embed'].copy()))
    new_batch['wav_mix'] = torch.concat(wav_mix)
    new_batch['wav_targets'] = torch.concat(wav_targets)
    new_batch['spk_embeds'] = torch.concat(spk_embeds)
    return new_batch


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True, repeat_dataset=False):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def Dataset(data_type,
            data_list_file,
            configs,
            spk2embed_dict=None,
            whole_utt=False,
            repeat_dataset=True,
            sample_random_spk=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            :param data_type(str): shard/raw/feat
            :param data_list_file: data list file
            :param configs: dataset configs
            :param spk2id_dict: spk2id dict
            :param reverb_lmdb_file: reverb data source lmdb file
            :param noise_lmdb_file: noise data source lmdb file
            :param whole_utt: use whole utt or random chunk
            :param repeat_dataset:
    """
    assert data_type in ['shard', 'raw']
    lists = read_lists(data_list_file)
    shuffle = configs.get('shuffle', False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle, repeat_dataset=repeat_dataset)
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    # Local shuffle
    if shuffle:
        dataset = Processor(dataset, processor.shuffle, **configs['shuffle_args'])

    # resample
    resample_rate = configs.get('resample_rate', 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)

    if not whole_utt:
        # random chunk
        chunk_len = configs.get('chunk_len', resample_rate * 3)
        dataset = Processor(dataset, processor.random_chunk, chunk_len)

    if sample_random_spk:
        dataset = Processor(dataset, processor.sample_spk_embedding, spk2embed_dict)
    # else:
    #     dataset = Processor(dataset, processor.get_spk_embedding, spk2embed_dict)

    return dataset
