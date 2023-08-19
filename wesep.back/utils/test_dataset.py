from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.dataset import tse_collate_fn_2spk
from wesep.utils.file_utils import load_speaker_embeddings

configs = {"shuffle": False, "resample_rate": 16000, "chunk_len": 32000}

spk2embed_dict = load_speaker_embeddings("data/clean/train-100/embed.scp", "data/clean/train-100/single.utt2spk")

dataset = Dataset('shard', 'data/clean/train-100/shard.list', configs=configs, spk2embed_dict=spk2embed_dict,
                  whole_utt=False,
                  repeat_dataset=False)

dataloader = DataLoader(dataset, batch_size=4, num_workers=1, collate_fn=tse_collate_fn_2spk)

for i, batch in enumerate(dataloader):
    print(batch['wav_mix'].size(), batch['wav_targets'].size(), batch['spk_embeds'].size())
    if i == 0:
        break
