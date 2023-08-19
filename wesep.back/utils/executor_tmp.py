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

import torch
import torchnet as tnt


def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
              scheduler,
              epoch,
              logger,
              scaler,
              enable_amp,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()

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
            outputs = model(features, enroll)  # (embed_a,embed_b) in most cases
            loss = criterion(outputs, targets).mean()

        # loss, acc
        loss_meter.add(loss.item())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % log_batch_interval == 0:
            logger.info("Epoch [%d], Iter [%d], Loss: %.4f" % (epoch, i + 1, loss.data.item()))

        if (i + 1) == epoch_iter:
            break

    logger.info("Epoch [%d], Iter [%d], Loss: %.4f" % (epoch, i + 1, loss.data.item()))
