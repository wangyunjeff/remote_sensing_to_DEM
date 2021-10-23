import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter


from models.gxynet import GxyNet
from data.build import Remote2DEMDataset
from engine.trainer import do_train
# -------------------------------#
# 超参数全部放在这里
# 方便回看和调整
# -------------------------------#
EXPERIMENT_TITLE = '1'

INIT_EPOCH = 0
FREEZE_EPOCH = 2000
UNFREEZE_EPOCH = 10000

LR = 1e-3
BATCH_SIZE = 4
CUDA = True

input_shape = (256,256)
num_classes = 1
num_workers = 0
# -------------------------------#
# 加载数据集
# -------------------------------#
transform_train = T.Compose([T.ToTensor()])
transform_val = T.Compose([T.ToTensor()])
dataset_root = '../data/datasets/remote2DEM_dataset'
with open(os.path.join(dataset_root, "ImageSets/train.txt"), "r") as f:
    train_lines = f.readlines()
with open(os.path.join(dataset_root, "ImageSets/val.txt"), "r") as f:
    val_lines = f.readlines()
train_dataset = Remote2DEMDataset(train_lines, input_shape, num_classes, dataset_root, transform_train)
val_dataset = Remote2DEMDataset(val_lines, input_shape, num_classes, dataset_root, transform_val)
gen_train = DataLoader(train_dataset,
                       shuffle=True,
                       batch_size=BATCH_SIZE,
                       num_workers=num_workers,
                       pin_memory=True,
                       drop_last=True)
gen_val = DataLoader(val_dataset,
                     shuffle=True,
                     batch_size=BATCH_SIZE,
                     num_workers=num_workers,
                     pin_memory=True,
                     drop_last=True)
# -------------------------------#
# 加载模型和训练
# -------------------------------#
model = GxyNet(num_classes=1, in_channels=4)
if CUDA:
    model = model.to("cuda")
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
loss_function = nn.MSELoss()

writer = SummaryWriter(os.path.join('logs', EXPERIMENT_TITLE))
for epoch in range(INIT_EPOCH, FREEZE_EPOCH):
    do_train(model=model,
             gen_train=gen_train,
             gen_val=gen_val,
             optimizer=optimizer,
             loss_function=loss_function,
             epoch=epoch,
             tensorboard=writer,
             cuda=CUDA)
    lr_scheduler.step()
# -------------------------------#
# 加载模型
# -------------------------------#



