import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data.dataset import Dataset


class Remote2DEMDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, dataset_path, transform=None, train=False):
        super(Remote2DEMDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transform = transform
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        # -------------------------------#
        # 从文件中读取图像
        # -------------------------------#
        remote = Image.open(os.path.join(os.path.join(self.dataset_path, "RemoteSensingImage"), name + ".png"))
        dem = Image.open(os.path.join(os.path.join(self.dataset_path, "DEM"), name + ".tif"))
        # -------------------------------#
        #   数据预处理和数据增强
        # -------------------------------#
        # remote = preprocess_input(np.array(remote, np.float64))
        # dem = np.array(dem)

        if self.transform is not None:
            remote = self.transform(remote)
            dem = self.transform(dem)

        return remote, dem