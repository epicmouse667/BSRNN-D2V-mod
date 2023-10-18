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
import torch.distributed as dist
import torch.nn.functional as F


class JointExecutor:

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
        ctc_losses = []
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                processed_wav_mix = batch['processed_wav_mix']
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
                processed_wav_mix = processed_wav_mix.float().to(device)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                labels = labels.int().to(device)
                label_lens = label_lens.int().to(device)
                wav_lens = wav_lens.int().to(device)
                prob_lens = prob_lens.int().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    logits,outputs = model(
                        wav_inputs = features,
                        processed_wav_mix = processed_wav_mix,
                        spk_embeddings = enroll
                    )
                    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

                    loss,si_snr,ctc_loss = criterion(
                        est_wav = outputs,
                        target_wav = targets,
                        wav_len = wav_lens,
                        log_prob = log_probs,
                        prob_len = prob_lens,
                        label = labels,
                        label_len = label_lens,
                    )
                    loss,si_snr,ctc_loss = loss.mean()/accum_iter,si_snr.mean()/accum_iter,ctc_loss.mean()/accum_iter
                loss.detach().cpu()
                si_snr.detach().cpu()
                ctc_loss.detach().cpu()
                # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                # dist.all_reduce(si_snr, op=dist.ReduceOp.SUM)
                # world_size = torch.cuda.device_count()
                # loss,si_snr = loss/world_size,si_snr/world_size
                losses.append(loss.item())
                si_snr_losses.append(si_snr.item())
                ctc_losses.append(ctc_loss.item())
                total_loss_avg = sum(losses) / len(losses)
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                total_ctc_loss = sum(ctc_losses)/len(ctc_losses)
                
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                if ((i + 1) % accum_iter == 0):
                    # updata the model
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                if (i + 1) % log_interval == 0 and rank ==0:
                    logger.info(
                        tp.row(("TRAIN", epoch, i + 1, total_loss_avg * accum_iter, total_si_snr_avg * accum_iter,
                                total_ctc_loss,
                                optimizer.param_groups[0]['lr']),
                               width=10,
                               style='grid'))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg * accum_iter,total_si_snr_avg * accum_iter

    def cv(self, dataloader, model, val_iter, criterion, epoch, enable_amp, logger,rank,
           log_batch_interval=100,
           device=torch.device('cuda')):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = log_batch_interval
        losses = []
        si_snr_losses = []
        ctc_losses = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                # processed_wav_mix = batch['processed_wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']
                labels = batch['labels']
                label_lens = batch['label_lens']
                wav_lens = batch['wav_lens']
                prob_lens = model.asr_model._get_feat_extract_output_lengths(
                    wav_lens
                ).to(torch.int32)

                features = features.float().to(device)  # (B,T,F)
                # processed_wav_mix = processed_wav_mix.float().to(device)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                labels = labels.int().to(device)
                label_lens = label_lens.int().to(device)
                wav_lens = wav_lens.int().to(device)
                prob_lens = prob_lens.int().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    logits,outputs = model(
                        wav_inputs = features,
                        processed_wav_mix = None, 
                        spk_embeddings = enroll
                    )
                    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    loss,si_snr,ctc_loss = criterion(
                        est_wav = outputs,
                        target_wav = targets,
                        wav_len = wav_lens,
                        log_prob = log_probs,
                        prob_len = prob_lens,
                        label = labels,
                        label_len = label_lens,
                    )
                    loss , si_snr , ctc_loss  = loss.mean() , si_snr.mean(), ctc_loss.mean()
                loss.detach().cpu()
                si_snr.detach().cpu()
                ctc_loss.detach().cpu()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(si_snr, op=dist.ReduceOp.SUM)
                dist.all_reduce(ctc_loss, op=dist.ReduceOp.SUM)
                world_size = torch.cuda.device_count()
                loss,si_snr,ctc_loss = loss/world_size,si_snr/world_size,ctc_loss/world_size
                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)
                si_snr_losses.append(si_snr.item())
                ctc_losses.append(ctc_loss.item())
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                total_ctc_loss_avg = sum(ctc_losses)/len(ctc_losses)
                if (i + 1) % log_interval == 0 and rank==0: 
                    logger.info(
                        tp.row(("VAL", epoch, i + 1, total_loss_avg, total_si_snr_avg,
                               total_ctc_loss_avg,
                                 '-'),
                               width=10,
                               style='grid'))
                if (i + 1) == val_iter:
                    break
        return total_loss_avg,total_si_snr_avg


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
                processed_wav_mix = batch['processed_wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']
                wav_lens = batch['wav_lens']

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                features = features.float().to(device)  # (B,T,F)
                processed_wav_mix = processed_wav_mix.float().to(device)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                wav_lens = wav_lens.int().to(device)
                with torch.cuda.amp.autocast(enabled=enable_amp):
                    # model.reset_param_recur()
                    est_wavs = model(
                        wav_inputs = features,
                        processed_wav_mix = processed_wav_mix, 
                        spk_embeddings = enroll
                    )
                    T_est,T_ref = est_wavs.shape[-1],targets.shape[-1]
                    if T_est > T_ref:
                        est_wavs = est_wavs[:,:T_ref]
                    elif T_ref > T_est:
                        est_wavs = F.pad(est_wavs,(0,T_ref-T_est))
                    assert est_wavs.shape == targets.shape ,(est_wavs.shape,targets.shape)
                    # print('est_wavs: ',est_wavs,'targets: ',targets)
                    si_snr = criterion(
                       ref = targets,
                       est = est_wavs,
                    )
                    si_snr = si_snr.mean()
                si_snr.detach().cpu()
                # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                # dist.all_reduce(si_snr, op=dist.ReduceOp.SUM)
                # world_size = torch.cuda.device_count()
                # loss,si_snr = loss/world_size,si_snr/world_size
                si_snr_losses.append(si_snr.item())
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                
                # scaler does nothing here if enable_amp=False
                scaler.scale(si_snr).backward()
                if ((i + 1) % accum_iter == 0):
                    # updata the model
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                if (i + 1) % log_interval == 0 and rank ==0:
                    logger.info(
                        tp.row(("TRAIN", epoch, i + 1, '-', total_si_snr_avg * accum_iter,
                                '-',
                                optimizer.param_groups[0]['lr']),
                               width=10,
                               style='grid'))
                if (i + 1) == epoch_iter:
                    break
            # scaler.step(optimizer)
            # optimizer.zero_grad()
            # scaler.update()
            total_loss_avg = sum(losses) / len(losses)
            return total_si_snr_avg * accum_iter

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
                processed_wav_mix = batch['processed_wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']
                wav_lens = batch['wav_lens']
                features = features.float().to(device)  # (B,T,F)
                processed_wav_mix = processed_wav_mix.float().to(device)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)
                wav_lens = wav_lens.int().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    est_wavs = model(
                        wav_inputs = features,
                        processed_wav_mix = processed_wav_mix, 
                        spk_embeddings = enroll
                    )
                    T_est,T_ref = est_wavs.shape[-1],targets.shape[-1]
                    if T_est > T_ref:
                        est_wavs = est_wavs[:,:T_ref]
                    elif T_ref > T_est:
                        est_wavs = F.pad(est_wavs,(0,T_ref-T_est))
                    assert est_wavs.shape == targets.shape ,(est_wavs.shape,targets.shape)
                    si_snr = criterion(
                       ref = targets,
                       est = est_wavs,
                    )
                    si_snr = si_snr.mean()
                si_snr.detach().cpu()
                # dist.all_reduce(si_snr, op=dist.ReduceOp.SUM)
                # si_snr = loss/world_size
                si_snr_losses.append(si_snr.item())
                total_si_snr_avg = sum(si_snr_losses)/len(si_snr_losses)
                if (i + 1) % log_interval == 0 and rank==0: 
                    logger.info(
                        tp.row(("VAL", epoch, i + 1, '-', total_si_snr_avg,
                                 '-',
                                 '-'),
                               width=10,
                               style='grid'))
                if (i + 1) == val_iter:
                    break
        return total_si_snr_avg