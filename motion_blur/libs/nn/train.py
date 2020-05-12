import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
from motion_blur.libs.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint, weighted_mse_loss
from motion_blur.libs.configs.read_config import parse_config
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def train(net, path_to_config: str = "motion_blur/libs/configs/config_motionnet.yml"):

    # Configs
    config = parse_config(path_to_config)

    # GPU
    if torch.cuda.is_available():
        net.to(device=torch.device("cuda"))

    # Initlialization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    running_loss = 0.0
    img = cv2.imread(config.train_dataset_path, 0)

    Path(config.save_path).mkdir(parents=True, exist_ok=True)

    ckp_path = Path(config.save_path) / "ckp.pth"
    save_path = Path(config.save_path) / "final_model.pth"

    # Resume
    if ckp_path.exists():
        start = load_checkpoint(ckp_path, net, optimizer)
    else:
        start = 0

    # weights
    weights = torch.tensor([1, 1]).type(torch.cuda.FloatTensor)

    # Training loop
    for epoch in range(start, config.n_epoch):

        # Mini batch
        mini_batch = torch.empty(config.mini_batch_size, 1, 512, 512).type(torch.cuda.FloatTensor)
        gt = torch.empty(config.mini_batch_size, 2).type(torch.cuda.FloatTensor)
        for i, x in enumerate(range(config.mini_batch_size)):
            # Randomly sample kernel
            L = config.L_min + torch.rand(1) * config.L_max
            theta = torch.rand(1) * 180
            kernel = motion_kernel(theta, L)
            H = Convolution(kernel)

            # Blur image
            image = H * img
            image = torch.tensor(image).type(torch.cuda.FloatTensor)
            # image = image / image.sum()

            mini_batch[i, 0, :, :] = image
            gt[i, 0] = torch.tensor(theta).type(torch.cuda.FloatTensor)
            gt[i, 1] = torch.tensor(L).type(torch.cuda.FloatTensor)

        # mini_batch.to(device = torch.device("cuda"))
        # gt.to(device = torch.device("cuda"))

        # GPU
        net.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        x = net.forward(mini_batch)
        # Backward pass
        # gt = torch.tensor(theta)[None, :].type(torch.cuda.FloatTensor)
        loss = weighted_mse_loss(x, gt, weights)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        if epoch % config.loss_period == (config.loss_period - 1):
            # Loss
            print("[%d, %5d] loss: %.3f" % (epoch + 1, config.n_epoch, running_loss / 2000))
            running_loss = 0.0

            # Checkpoint
            ckp = define_checkpoint(net, optimizer, epoch)
            save_checkpoint(ckp, ckp_path)

    torch.save(net.state_dict(), save_path)
