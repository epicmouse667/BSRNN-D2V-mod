from torch.utils.data import DataLoader
import torch
from wesep.dataset.dataset import Dataset
from wesep.utils.file_utils import load_speaker_embeddings

configs = {"shuffle": False, "resample_rate": 16000, "chunk_len": 32000}

spk2embed_dict = load_speaker_embeddings("data/clean/test/embed.scp", "data/clean/test/single.utt2spk")

dataset = Dataset('shard', 'data/clean/test/shard.list', configs=configs, spk2embed_dict=spk2embed_dict,
                  whole_utt=False,
                  repeat_dataset=False)


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


dataloader = DataLoader(dataset, batch_size=4, num_workers=1, collate_fn=tse_collate_fn_2spk)

for i, batch in enumerate(dataloader):
    print(batch['wav_mix'].size(), batch['wav_targets'].size(), batch['spk_embeds'].size())
    if i == 0:
        break
