import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution


class MotionNet(nn.Module):
    def __init__(self):

        super(MotionNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 3)
        # nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(8, 16, 3)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 32, 3)
        # nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(32, 64, 3)
        # nn.init.xavier_uniform_(self.conv4.weight)
        # self.adaptive_pool = nn.AdaptiveMaxPool2d((32, 32))
        self.adaptive_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(30 * 30 * 64, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x: torch.tensor):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(F.relu(self.conv4(x)))
        x = x.view(-1, 30 * 30 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
