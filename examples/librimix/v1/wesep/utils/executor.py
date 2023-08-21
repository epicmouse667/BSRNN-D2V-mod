# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp
# if your python version < 3.7 use the below one
import torch
from contextlib import nullcontext


class Executor:

    def __init__(self):
        self.step = 0

    def train(self, dataloader, model, epoch_iter, optimizer, criterion, scheduler, scaler, epoch, enable_amp, logger,rank,
              accum_iter=1,log_batch_interval=100,
              device=torch.device('cuda')):
        ''' Train one epoch
                '''
        model.train()
        log_interval = log_batch_interval
        losses = []
        si_snr_losses = []
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']
                labels = batch['labels']
                label_lens = batch['label_lens']
                wav_lens = batch['wav_lens']
                prob_lens = model.asr_model._get_feat_extract_output_lengths(
                    wav_lens
                ).to(torch.int32)

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                labels = labels.int().to(device)
                label_lens = label_lens.int().to(device)
                wav_lens = wav_lens.int().to(device)
                prob_lens = prob_lens.int().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    log_probs,outputs = model(
                        wav_inputs = features, 
                        spk_embeddings = enroll)
                    loss,si_snr = criterion(
                        est_wav = outputs,
                        target_wav = targets,
                        wav_len = wav_lens,
                        log_prob = log_probs,
                        prob_len = prob_lens,
                        label = labels,
                        label_len = label_lens,
                    )
                    loss,si_snr = loss.mean()/accum_iter,si_snr.mean()/accum_iter

                losses.append(loss.item())
                si_snr_losses.append(si_snr.item())
                total_loss_avg = sum(losses) / len(losses)
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                if ((i + 1) % accum_iter == 0) or (i + 1 == len(dataloader)):
                    # updata the model
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                loss.detach().cpu()
                si_snr.detach().cpu()
                if (i + 1) % log_interval == 0 and rank==0:
                    logger.info(
                        tp.row(("TRAIN", epoch, i + 1, total_loss_avg * accum_iter, total_si_snr_avg * accum_iter,optimizer.param_groups[0]['lr']),
                               width=10,
                               style='grid'))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg,total_si_snr_avg

    def cv(self, dataloader, model, val_iter, criterion, epoch, enable_amp, logger,rank,
           log_batch_interval=100,
           device=torch.device('cuda')):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = log_batch_interval
        losses = []
        si_snr_losses = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']
                labels = batch['labels']
                label_lens = batch['label_lens']
                wav_lens = batch['wav_lens']
                prob_lens = model.asr_model._get_feat_extract_output_lengths(
                    wav_lens
                ).to(torch.int32)

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                labels = labels.int().to(device)
                label_lens = label_lens.int().to(device)
                wav_lens = wav_lens.int().to(device)
                prob_lens = prob_lens.int().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    log_probs,outputs = model(
                        wav_inputs = features, 
                        spk_embeddings = enroll)
                    loss,si_snr = criterion(
                        est_wav = outputs,
                        target_wav = targets,
                        wav_len = wav_lens,
                        log_prob = log_probs,
                        prob_len = prob_lens,
                        label = labels,
                        label_len = label_lens,
                    )
                    loss , si_snr = loss.mean() , si_snr.mean()
                loss.detach().cpu()
                si_snr.detach().cpu()
                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)
                si_snr_losses.append(si_snr.item())
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                if (i + 1) % log_interval == 0 and rank==0: 
                    logger.info(
                        tp.row(("VAL", epoch, i + 1, total_loss_avg, total_si_snr_avg,'-'),
                               width=10,
                               style='grid'))
                if (i + 1) == val_iter:
                    break
        return total_loss_avg,total_si_snr_avg
