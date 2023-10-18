from __future__ import print_function

import os
import time
import fire
import soundfile
import torch
from torch.utils.data import DataLoader
from wesep.models import get_model

from wesep.dataset.dataset import Dataset, tse_collate_fn_2spk
from wesep.utils.file_utils import read_label_file, read_vec_scp_file
from wesep.utils.score import cal_SISNRi,cal_WER
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed
from transformers import Data2VecAudioForCTC,AutoProcessor

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def infer(config='confs/conf.yaml', **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_wer = 0
    total_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    rank = 0
    set_seed(configs['seed'] + rank)
    gpu = configs['gpus']
    device = torch.device(
        "cuda:{}".format(gpu)) if gpu >= 0 else torch.device("cpu")

    model = get_model(configs['model'])(**configs['model_args'])
    model_path = os.path.join(configs['checkpoint'])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    logger = get_logger(configs['exp_dir'], 'infer1.log')
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs['exp_dir'], 'audio')
    model = model.to(device)
    model.eval()

    test_spk_embeds = configs['test_spk_embeds']
    test_spk1_embed_scp = configs['test_spk1_enroll']
    test_spk2_embed_scp = configs['test_spk2_enroll']
    test_spk2embed_dict = read_vec_scp_file(test_spk_embeds)

    test_spk1_embed = read_label_file(test_spk1_embed_scp)
    test_spk2_embed = read_label_file(test_spk2_embed_scp)

    with open(test_spk_embeds, 'r') as f:
        lines = f.readlines()

    test_dataset = Dataset(configs['data_type'],
                           configs['test_data'],
                           configs['dataset_args'],
                           test_spk2embed_dict,
                           test_spk1_embed,
                           test_spk2_embed,
                           state='test',
                           whole_utt=configs.get('whole_utt', True),
                           repeat_dataset=configs.get('repeat_dataset', False),
                           reverb=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=tse_collate_fn_2spk)
    test_iter = len(lines) // 2
    logger.info('test number: {}'.format(test_iter))

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            features = batch['wav_mix']
            targets = batch['wav_targets']
            enroll = batch['spk_embeds']
            spk = batch['spk']
            key = batch['key']
            transes = batch['transes']

            features = features.float().to(device)  # (B,T,F)
            targets = targets.float().to(device)
            enroll = enroll.float().to(device)

            logits,outputs = model(features,None,enroll)
            processor = AutoProcessor.from_pretrained(configs['model_args']['pretrained_d2v4ctc_dir'])
            # outputs =torch.cat([processor(x,sampling_rate=model.sr,return_tensors="pt")['input_values'] for x in targets],dim=0) 
            predicted_ids = torch.argmax(logits, dim=-1)
            est_trans = processor.batch_decode(predicted_ids)
            

            ref = targets.cpu().numpy()
            ests = outputs.cpu().numpy()
            mix = features.cpu().numpy()

            if ests[0].size != ref[0].size:
                end = min(ests[0].size, ref[0].size, mix[0].size)
                ests_1 = ests[0][:end]
                ref_1 = ref[0][:end]
                mix_1 = mix[0][:end]
                SISNR1, delta1 = cal_SISNRi(ests_1, ref_1, mix_1)
            else:
                SISNR1, delta1 = cal_SISNRi(ests[0], ref[0], mix[0])
            wer1 = cal_WER(est=est_trans[0],ref=transes[0])

            logger.info(
                "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f} | WER={:2f}".format(total_cnt + 1, key[0],
                                                                                              spk[0], SISNR1, delta1,wer1))
            total_SISNR += SISNR1
            total_SISNRi += delta1
            total_wer += wer1
            total_cnt += 1

            if ests[1].size != ref[1].size:
                end = min(ests[1].size, ref[1].size, mix[1].size)
                ests_2 = ests[1][:end]
                ref_2 = ref[1][:end]
                mix_2 = mix[1][:end]
                SISNR2, delta2 = cal_SISNRi(ests_2, ref_2, mix_2)
            else:
                SISNR2, delta2 = cal_SISNRi(ests[1], ref[1], mix[1])
            wer2 = cal_WER(est=est_trans[1],ref=transes[1])
            logger.info(
                "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f} | WER={:2f}".format(total_cnt + 1, key[1],
                                                                                              spk[1], SISNR2, delta2,wer2))
            total_SISNR += SISNR2
            total_SISNRi += delta2
            total_wer += wer2
            total_cnt += 1

            # if (i + 1) == test_iter:
            #     break
        end = time.time()

    logger.info('Time Elapsed: {:.1f}s'.format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info("Average WER: {:.2f}".format(total_wer / total_cnt))


if __name__ == '__main__':
    fire.Fire(infer)