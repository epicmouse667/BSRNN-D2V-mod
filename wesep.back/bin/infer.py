from __future__ import print_function

import os
import time

import soundfile
import torch
from wesep.models.speaker_model import get_speaker_model

from wesep.utils.file_utils import read_vec_scp_file, read_label_file
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed
from wesep.dataset.dataset import Dataset,tse_collate_fn_2spk
from wesep.utils.metrics import cal_SISNRi

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def infer(config='/mntcephfs/lab_data/wangxuefei/works/wesep/examples/librimix/v1/confs/infer_convtasnet.yaml',
          **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    rank = 0
    set_seed(configs['seed'] + rank)
    device = torch.device("cpu")

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    model_path = os.path.join(configs['checkpoint'])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    logger = get_logger(configs['exp_dir'], 'infer.log')
    logger.info("Load checkpoint from {}".format(model_path))

    model = model.to(device)
    model.eval()

    test_spk_embeds = configs['test_spk_embeds']
    test_single_utt2spk = configs['test_single_utt2spk']
    test_spk1_embed_scp = configs['test_spk1_embed']
    test_spk2_embed_scp = configs['test_spk2_embed']
    test_spk2embed_dict = read_vec_scp_file(test_spk_embeds)

    test_spk1_embed = read_label_file(test_spk1_embed_scp)
    test_spk2_embed = read_label_file(test_spk2_embed_scp)

    with open(test_single_utt2spk, 'r') as f:
        lines = f.readlines()

    test_dataset = Dataset(configs['data_type'],
                           configs['test_data'],
                           configs['dataset_args'],
                           test_spk2embed_dict,
                           test_spk1_embed,
                           test_spk2_embed,
                           state='test',
                           whole_utt=configs.get('whole_utt', True),
                           repeat_dataset=configs.get('repeat_dataset', True))
    test_dataloader = DataLoader(test_dataset, **configs['dataloader_args'], collate_fn=tse_collate_fn_2spk)
    test_iter = len(lines)
    logger.info('test number: {}'.format(test_iter))

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            features = batch['wav_mix']
            targets = batch['wav_targets']
            enroll = batch['spk_embeds']

            features = features.float().to(device)  # (B,T,F)
            targets = targets.float().to(device)
            enroll = enroll.float().to(device)

            outputs = model(features, enroll)

            utt_name = total_cnt + 1
            soundfile.write(f'/mntcephfs/lab_data/wangxuefei/works/wesep/examples/librimix/v1/audio/{utt_name}.wav',
                            outputs[0], 16000)

            ref = targets.cpu().numpy()
            ests = outputs.cpu().numpy()
            mix = features.cpu().numpy()

            if ests.size != ref.size:
                end = min(ests.size, ref.size)
                ests = ests[:end]
                ref = ref[:end]
                mix = mix[:end]

            # for each utts
            # Compute SI-SNR
            SISNR, delta = cal_SISNRi(ests, ref, mix)

            logger.info(
                "Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f}".format(
                    total_cnt + 1, SISNR, delta))
            total_SISNR += SISNR
            total_SISNRi += delta
            total_cnt += 1
        end = time.time()

    logger.info('Time Elapsed: {:.1f}s'.format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))


if __name__ == '__main__':
    infer()
