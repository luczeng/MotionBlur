# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import cv2
from motion_blur.libs.nn.motion_net import MotionNet
from motion_blur.libs.forward_models.functions import *
from motion_blur.libs.nn.train import train
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
from motion_blur.libs.configs.read_config import parse_config
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs the motion blur estimation network on a blurred image with a randomly generated kernel"
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="motion_blur/libs/configs/config_motionnet.yml",
        help="Path to config file describing net format, weight path and input image path",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Change those if you want to visualize different effects
    L_user = 10
    Theta_user = 45

    # Configs
    args = parse_args()
    config = parse_config(args.config_path)

    path_to_model = Path(config.weight_path) / "final_model.pth"

    # Load model
    net = MotionNet()
    net.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))

    # Generate inferences
    theta_vec = torch.linspace(0, 180, 50)
    L_vec = torch.linspace(1, 20, 50)
    theta_infered, L_infered, theta_true, L_true = [], [], [], []
    for theta in theta_vec:
        L = torch.tensor(L_user)
        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)

        # Blur image
        img = cv2.imread(config.test_dataset_path, 0)
        image = H * img
        image = torch.tensor(image[None, None, :, :]).to(dtype=torch.float32)

        # GPU
        net.zero_grad()

        # Run inference
        x = net.forward(image)

        theta_infered.append(x[0][0].cpu().detach().numpy())
        theta_true.append(theta.cpu().detach().numpy())

    for L in L_vec:
        theta = torch.tensor(Theta_user)
        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)

        # Blur image
        img = cv2.imread(config.test_dataset_path, 0)
        image = H * img
        image = torch.tensor(image[None, None, :, :]).to(dtype=torch.float32)

        # GPU
        net.zero_grad()

        # Run inference
        x = net.forward(image)

        L_infered.append(x[0][1].cpu().detach().detach().numpy())
        L_true.append(L.cpu().detach().numpy())

    plt.plot(theta_infered, "xr", label="Infered angle")
    plt.plot(theta_true, "xb", label="True angle")
    plt.xlabel("draw number")
    plt.ylabel("Angle (°)")
    plt.tick_params(axis="x", bottom=False)
    plt.legend()
    plt.title("Angle inference for fixed motion blur length (L = {})".format(L_user))
    plt.savefig("imgs/linear_motion_blur_perfomance_theta.png")
    plt.show()
    plt.clf()

    plt.plot(L_infered, "xr", label="Infered length")
    plt.plot(L_true, "xb", label="True length")
    plt.xlabel("draw number")
    plt.ylabel("Blur length (pixel)")
    plt.tick_params(axis="x", bottom=False)
    plt.legend()
    plt.title("Blur length inference for fixed angle (theta = {})".format(Theta_user))
    plt.savefig("imgs/linear_motion_blur_perfomance_length.png")
    plt.show()
    plt.clf()
