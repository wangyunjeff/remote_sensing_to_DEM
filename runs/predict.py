import os

import torch
from utils.utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import config.config as config
from data.build import  Remote2DEMDataset
from models.unet import Unet



torch.backends.cudnn.benchmark = True


epoch = 0
model = Unet(num_classes=1, in_channels=3).to(config.DEVICE)

opt_gen = optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.999))
load_checkpoint(
            config.CHECKPOINRT_PATH, model, opt_gen, 0.1,
        )

with open(os.path.join(config.DATASET_ROOT, "ImageSets/train.txt"), "r") as f:
    val_lines = f.readlines()
val_dataset = Remote2DEMDataset(val_lines, (256, 256), 3, config.DATASET_ROOT, config.transform_input, config.transform_output)
val_loader = DataLoader(val_dataset, batch_size=9, shuffle=False, drop_last=True)

save_some_examples(model, val_loader, epoch, folder="../evaluation")