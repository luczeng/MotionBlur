import torch.nn as nn
import torch.nn.functional as F


class MotionNet(nn.Module):
    def __init__(self):
        """
            Network layer definition
            Note: architecture is currently evolving a lot
            TODO: evaluate which best init is best
        """

        super(MotionNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv12 = nn.Conv2d(8, 8, 3)

        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv22 = nn.Conv2d(16, 16, 3)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv32 = nn.Conv2d(32, 32, 3)

        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv42 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(199424, 256)
        self.fc2 = nn.Linear(256, 2)

    def _one_pass(self, x):
        """
            Pass of the net on one image
            :param x input tensor
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv42(x))
        x = self.pool(x)

        x = x.view(-1, 199424)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, input):
        """
            Pass of the net over a batch
            :param input image tensor or list of tensors
        """

        if type(input) == list:
            scores = []
            for i, x in enumerate(input):
                score = self._one_pass(x)
                scores.append(score)
        else:
            scores = self._one_pass(input)

        return scores
