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

    def train(self, dataloader, model, epoch_iter, optimizer, criterion, scheduler, scaler, epoch, enable_amp, logger,
              log_batch_interval=100,
              device=torch.device('cuda')):
        ''' Train one epoch
                '''
        model.train()
        log_interval = log_batch_interval
        accum_grad = 1
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']

                cur_iter = (epoch - 1) * epoch_iter + i
                scheduler.step(cur_iter)

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll)
                    loss = criterion(outputs, targets).mean() / accum_grad

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # updata the model
                optimizer.zero_grad()
                # scaler does nothing here if enable_amp=False
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(("TRAIN", epoch, i + 1, total_loss_avg * accum_grad, optimizer.param_groups[0]['lr']),
                               width=10,
                               style='grid'))
                if (i + 1) == epoch_iter:
                    break
            total_loss_avg = sum(losses) / len(losses)
            return total_loss_avg

    def cv(self, dataloader, model, val_iter, criterion, epoch, enable_amp, logger,
           log_batch_interval=100,
           device=torch.device('cuda')):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = log_batch_interval
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                features = batch['wav_mix']
                targets = batch['wav_targets']
                enroll = batch['spk_embeds']

                features = features.float().to(device)  # (B,T,F)
                targets = targets.float().to(device)
                enroll = enroll.float().to(device)

                with torch.cuda.amp.autocast(enabled=enable_amp):
                    outputs = model(features, enroll)
                    loss = criterion(outputs, targets).mean()

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(("VAL", epoch, i + 1, total_loss_avg, '-'),
                               width=10,
                               style='grid'))
                if (i + 1) == val_iter:
                    break
        return total_loss_avg
