import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T

TIME = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())

# 实验记录设置
EXPERIMENT_TITLE = 'unet-remote2DEM'
EXPERIMENT_LOGS_DIR = '../runs/logs/{}/{}/'.format(EXPERIMENT_TITLE, TIME)

# 训练参数设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = '../data/datasets/remote2DEM_dataset'
# 超参数设置
INIT_EPOCH = 0
FREEZE_EPOCH = 2000
UNFREEZE_EPOCH = 10000

NUM_CLASSES = 1
IN_CHANNEL = 4
LR = 1e-3
BATCH_SIZE = 4
CUDA = True

SAVE_MODEL = True
LOAD_MODEL = True
CHECKPOINRT_PATH = r'G:\Projects\GitHub\remote_sensing_to_DEM\runs\logs\unet-remote2DEM\20211027_17-04-24\weight\checkpoint_epoch5.pth'

input_shape = (256, 256)
num_classes = 1
num_workers = 0

transform_train = T.Compose([T.ToTensor()])
transform_val = T.Compose([T.ToTensor()])
