import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

import config.config as config
from models.gxynet import GxyNet
from data.build import Remote2DEMDataset
from engine.trainer import do_train
from utils.utils import save_checkpoint, load_checkpoint

TIME = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
# -------------------------------#
# 超参数全部放在这里
# 方便回看和调整
# -------------------------------#


NUM_CLASSES = 1
IN_CHANNEL = 4


# -------------------------------#
# 加载数据集
# -------------------------------#

with open(os.path.join(config.DATASET_ROOT, "ImageSets/train.txt"), "r") as f:
    train_lines = f.readlines()
with open(os.path.join(config.DATASET_ROOT, "ImageSets/val.txt"), "r") as f:
    val_lines = f.readlines()

train_dataset = Remote2DEMDataset(train_lines, config.input_shape, config.num_classes, config.DATASET_ROOT, config.transform_train)
val_dataset = Remote2DEMDataset(val_lines, config.input_shape, config.num_classes, config.DATASET_ROOT, config.transform_val)
gen_train = DataLoader(train_dataset,
                       shuffle=True,
                       batch_size=config.BATCH_SIZE,
                       num_workers=config.num_workers,
                       pin_memory=True,
                       drop_last=True)
gen_val = DataLoader(val_dataset,
                     shuffle=True,
                     batch_size=config.BATCH_SIZE,
                     num_workers=config.num_workers,
                     pin_memory=True,
                     drop_last=True)
# -------------------------------#
# 加载模型和训练
# -------------------------------#

model = GxyNet(num_classes=1, in_channels=4).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=5e-4)

loss_function = nn.MSELoss()
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINRT_PATH, model, optimizer, config.LR)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
writer = SummaryWriter(os.path.join(config.EXPERIMENT_LOGS_DIR, 'tensorboard', ))
for epoch in range(config.INIT_EPOCH, config.FREEZE_EPOCH):
    do_train(
        model=model,
        gen_train=gen_train,
        gen_val=gen_val,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        tensorboard=writer,
    )

    if config.SAVE_MODEL and epoch % 5 == 0:
        save_checkpoint(model,
                        optimizer,
                        os.path.join(config.EXPERIMENT_LOGS_DIR, 'weight/checkpoint_epoch{}.pth'.format(epoch)))
    lr_scheduler.step()
# -------------------------------#
# 加载模型
# -------------------------------#
