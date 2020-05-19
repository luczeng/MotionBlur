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

    # Configs
    args = parse_args()
    config = parse_config(args.config_path)

    path_to_model = Path(config.weight_path) / "final_model.pth"

    # Load model
    net = MotionNet()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(path_to_model))
        net.to(device=torch.device("cuda"))
        net_type = torch.cuda.FloatTensor
    else:
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))
        net_type = torch.FloatTensor

    # Randomly sample kernel
    L = config.L_min + torch.rand(1) * config.L_max
    theta = torch.rand(1) * 180
    kernel = motion_kernel(theta, L)
    H = Convolution(kernel)

    # Blur image
    img = cv2.imread(config.test_dataset_path, 0)
    image = H * img
    image = torch.tensor(image[None, None, :, :]).type(net_type)

    # GPU
    net.zero_grad()

    # Run inference
    x = net.forward(image)

    print(
        "Estimated angle and length: {}\nTrue angle and length:      [{} {}].".format(
            x.to("cpu").detach().numpy(), theta.to("cpu").numpy()[0], L.to("cpu").numpy()[0]
        )
    )
