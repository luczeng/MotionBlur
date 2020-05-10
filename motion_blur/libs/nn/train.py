import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import cv2
import matplotlib.pyplot as plt


def train(net, path_to_config: str = "motion_blur/libs/configs/config_motionnet.yml"):

    # Configs
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initlialization
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config["TRAIN"]["LR"])
    running_loss = 0.0
    img = cv2.imread(config["TRAIN"]["DATASET_PATH"], 0)

    for epoch in range(config["TRAIN"]["N_EPOCH"]):

        # Randomly sample kernel
        L = config["TRAIN"]["L_min"] + torch.rand(1) * config["TRAIN"]["L_max"]
        theta = torch.rand(1) * 180
        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)

        # Blur image
        image = H * img
        image = torch.tensor(image[None, None, :, :]).float()

        net.zero_grad()
        optimizer.zero_grad()

        x = net.forward(image)
        x = x / x.sum()
        loss = criterion(x, torch.tensor(theta)[None, :])
        loss.backward()
        # print(x.detach().numpy())

        optimizer.step()
        running_loss += loss.item()
        if epoch % 100 == 99:
            # Loss
            print("[%d, %5d] loss: %.3f" % (epoch + 1, config["TRAIN"]["N_EPOCH"], running_loss / 2000))
            running_loss = 0.0
            # Checkpoint
            # ckp = utils.define_checkpoint(net, optimizer, i)
            # utils.save_checkpoint(ckp, ckp_path)
