import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution


class MotionNet(nn.Module):
    def __init__(self):
        """
            Network layer definition
            TODO: evaluate which best init is best
        """

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
        self.adaptive_pool = nn.AdaptiveMaxPool2d((28, 28))
        # self.adaptive_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(28 * 28 * 64, 256)
        self.fc2 = nn.Linear(256, 2)

    def one_pass(self, x):
        """
            Pass of the net on one image
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(F.relu(self.conv4(x)))
        x = x.view(-1, 28 * 28 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, input):
        """
            Pass of the net over a batch
        """

        scores = []
        for i, x in enumerate(input):
            score = self.one_pass(x)
            scores.append(score)

        return scores
