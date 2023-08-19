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

import os
import re
from pprint import pformat

import fire
import matplotlib.pyplot as plt
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wesep.utils.schedulers as schedulers
from wesep.dataset.dataset import Dataset, tse_collate_fn_2spk
from wesep.models import get_model
from wesep.utils.checkpoint import load_checkpoint, save_checkpoint
from wesep.utils.executor import run_epoch
from wesep.utils.file_utils import load_speaker_embeddings, read_label_file
from wesep.utils.metrics import SISNRLoss
from wesep.utils.utils import get_logger, parse_config_or_kwargs, set_seed


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print(model_dir + " already exists !!!")
            if checkpoint is None:
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # embeds
    tr_spk_embeds = configs['train_spk_embeds']
    tr_single_utt2spk = configs['train_utt2spk']
    tr_utt_spk_list = read_label_file(tr_single_utt2spk)
    tr_spk2embed_dict = load_speaker_embeddings(tr_spk_embeds, tr_single_utt2spk)

    val_spk_embeds = configs['val_spk_embeds']
    val_single_utt2spk = configs['val_utt2spk']
    val_spk2embed_dict = load_speaker_embeddings(val_spk_embeds, val_single_utt2spk)

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            tr_spk2embed_dict,
                            whole_utt=configs.get('whole_utt', False),
                            repeat_dataset=configs.get('repeat_dataset', True))
    val_dataset = Dataset(configs['data_type'],
                          configs['val_data'],
                          configs['dataset_args'],
                          val_spk2embed_dict,
                          whole_utt=configs.get('whole_utt', False),
                          repeat_dataset=configs.get('repeat_dataset', False))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'], collate_fn=tse_collate_fn_2spk)
    val_dataloader = DataLoader(val_dataset, **configs['dataloader_args'], collate_fn=tse_collate_fn_2spk)
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(tr_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model
    logger.info("<== Model ==>")
    model = get_model(configs['model'])(**configs['model_args'])
    num_params = sum(param.numel() for param in model.parameters())
    if rank == 0:
        logger.info('The model size: {}'.format(num_params))
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')

    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)

    # If specify checkpoint, load some info from checkpoint.
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    device = torch.device("cuda")

    criterion = SISNRLoss()
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=2, min_lr=1e-8)
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'iter', 'Learning Rate', 'Loss']
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        train_losses = []
        val_losses = []

        train_loss = run_epoch(train_dataloader,
                               epoch_iter,
                               ddp_model,
                               criterion,
                               optimizer,
                               scheduler,
                               epoch,
                               logger,
                               scaler,
                               enable_amp=configs['enable_amp'],
                               log_batch_interval=configs['log_batch_interval'],
                               device=device)

        # val_loss = run_epoch(val_dataloader,
        #                      epoch_iter,
        #                      ddp_model,
        #                      criterion,
        #                      optimizer,
        #                      scheduler,
        #                      epoch,
        #                      logger,
        #                      scaler,
        #                      enable_amp=configs['enable_amp'],
        #                      log_batch_interval=configs['log_batch_interval'],
        #                      device=device)
        train_losses.append(train_loss)
        # val_losses.append(val_loss)
        #
        # best_loss = val_loss
        scheduler.best = train_loss
        scheduler.step(train_loss)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch >= configs['num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))

    plt.title("Loss of train and dev")
    x = [i for i in range(configs['num_epochs'] + 1)]
    plt.plot(x, train_losses, 'b-', label=u'train_loss', linewidth=0.8)
    # plt.plot(x, val_losses, 'c-', label=u'val_loss', linewidth=0.8)
    plt.legend()
    # plt.xticks(l, lx)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('conv_tasnet_LRS.png')


if __name__ == '__main__':
    fire.Fire(train)
