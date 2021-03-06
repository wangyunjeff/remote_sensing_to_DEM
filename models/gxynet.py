import torch
import torch.nn as nn
from torchsummary import summary


class SematicEmbbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(SematicEmbbedBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.conv3x3(high_x))
        low_x = self.conv1x1(low_x)
        return high_x * low_x


class GxyNet(nn.Module):
    """
    downsample ratio=2
    """

    def __init__(self, num_classes=1, in_channels=3):
        super(GxyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(12, 20, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(20)
        self.relu3 = nn.ReLU(True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(20, 40, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(40)
        self.relu4 = nn.ReLU(True)

        self.seb1 = SematicEmbbedBlock(40, 20, 20)
        self.seb2 = SematicEmbbedBlock(20, 12, 12)
        self.seb3 = SematicEmbbedBlock(12, 6, 6)

        self.heatmap = nn.Conv2d(6, num_classes, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        m1 = self.maxpool1(x1)
        # print(m1.shape)

        x2 = self.conv2(m1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        m2 = self.maxpool2(x2)
        # print(m2.shape)

        x3 = self.conv3(m2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        m3 = self.maxpool3(x3)
        # print(m3.shape)

        x4 = self.conv4(m3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        # print(x4.shape)

        up1 = self.seb1(x4, x3)
        # print(up1.shape)
        up2 = self.seb2(up1, x2)
        # print(up2.shape)
        up3 = self.seb3(up2, x1)
        # print(up3.shape)

        out = self.heatmap(up3)
        return out


if __name__ == '__main__':
    model = GxyNet(num_classes=1, in_channels=3).to("cuda")
    summary(model, input_size=(3, 256, 256))
