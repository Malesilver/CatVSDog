import torch as t
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Alexnet(nn.Module):

    def __init__(self, in_channels, num_classes = 2):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.linear_input_size(in_channels), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def linear_input_size(self, in_channels):
        x = t.rand(BATCH_SIZE, in_channels, RE_HEIGHT, RE_WIDTH)
        return self.features(x).view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x

class Simplenet(nn.Module):

    def __init__(self, in_channels, num_classes = 2):
        super(Simplenet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.linear_input_size(in_channels), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def linear_input_size(self, in_channels):
        x = t.rand(BATCH_SIZE, in_channels, RE_HEIGHT, RE_WIDTH)
        return self.features(x).view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x

class Simplenet2(nn.Module):

    def __init__(self, in_channels, num_classes = 2):
        super(Simplenet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2, padding = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.linear_input_size(in_channels), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def linear_input_size(self, in_channels):
        x = t.rand(BATCH_SIZE, in_channels, RE_HEIGHT, RE_WIDTH)
        return self.features(x).view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x

