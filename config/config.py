import time
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T

TIME = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())

# 实验记录设置
EXPERIMENT_TITLE = 'unet-remote2DEM'  # 跑不同的实验时重新设置该项
EXPERIMENT_LOGS_DIR = '../runs/logs/{}/{}/'.format(EXPERIMENT_TITLE, TIME)

# 训练参数设置
DATASET_ROOT = '../data/datasets/remote2DEM_dataset_qinling'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数设置
NUM_EPOCHS = 2000
LR = 1e-3
BATCH_SIZE = 16
NUM_CLASSES = 1
IN_CHANNEL = 4


# 加载和保存模型
SAVE_MODEL = True
LOAD_MODEL = False
CHECKPOINRT_PATH = '../runs/logs/unet-remote2DEM/20211104_20-59-19/weight/checkpoint_epoch300.pth'

input_shape = (256, 256)
num_classes = 1
num_workers = 0

transform_input = T.Compose([T.ToTensor(),
                             T.Normalize(mean=[0.2895, 0.3111, 0.2108], std=[0.1522, 0.1278, 0.1191])
                             ])

transform_output = T.Compose([T.ToTensor(),
                           T.Normalize(mean=[2532.5225], std=[1147.7783])
                           ])
