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

from wesep.models.SI_SNR import si_snr_loss
from torch.nn.parallel import data_parallel

def to_device(dicts, device):
    '''
       load dict data to cuda
    '''
    
    def to_cuda(datas):
        # print(datas)
        if isinstance(datas, torch.Tensor):
            return datas.to(device, dtype=torch.float)
        elif isinstance(datas,list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')
    
def run_epoch(dataloader,
              epoch_iter,
              model,
              criterion,
              optimizer,
            #   scheduler,
            #   margin_scheduler,
              epoch,
              logger,
              scaler,
              enable_amp,
              stats,
              log_batch_interval=100,
              device=torch.device('cuda')
              ):
    if stats == 'train':
        model.train()
    # By default use average pooling
    # loss_meter = tnt.meter.AverageValueMeter()
    # acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    losses = []
    for i, egs in enumerate(dataloader):
        # print(egs)
        egs = to_device(egs, device)

        if stats == 'train':
            optimizer.zero_grad()
            # print(egs['mixture'])
            with torch.cuda.amp.autocast(enabled=False):
                ests = model(egs['mixture'])
            loss = si_snr_loss(ests, egs)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_loss_avg = sum(losses)/len(losses)

        elif stats == 'val':
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    ests = model(egs['mixture'])
                loss = si_snr_loss(ests, egs)
                # loss.backward()
                # optimizer.step()
                losses.append(loss.item())
                total_loss_avg = sum(losses)/len(losses) 


        # log
        if (i + 1) % log_batch_interval == 0:
            logger.info(
                tp.row((epoch, i + 1, optimizer.param_groups[0]['lr'], total_loss_avg, stats), 
                        width=10, style='grid')) 
            
        if (i + 1) == epoch_iter:
            break        

    logger.info(
        tp.row((epoch, i + 1, optimizer.param_groups[0]['lr'], total_loss_avg, stats), 
               width=10, style='grid'))
    
    return total_loss_avg

