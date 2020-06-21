from motion_blur.libs.nn.motion_net import MotionNet
from motion_blur.libs.configs.read_config import parse_config
from motion_blur.libs.nn.train import run_train
from motion_blur.libs.nn.train_small import run_train_small
from motion_blur.libs.utils.nn_utils import print_training_info
from motion_blur.libs.utils.nn_utils import log_mlflow_param
from pathlib import Path
import argparse
import torch
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()

    return args


def run_train(args):
    # Configs
    cfg = parse_config(args.config_path)

    # Path
    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    ckp_path = Path(cfg.save_path) / "ckp.pth"
    save_path = Path(cfg.save_path) / "final_model.pth"

    # Net
    reds_size = [720, 1280]
    net = MotionNet(
        cfg.n_layers,
        cfg.n_sublayers,
        cfg.n_features_first_layer,
        reds_size,
        cfg.as_gray,
        cfg.regression,
        cfg.n_angles,
        cfg.n_lengths,
    )
    # reds_size = (720, 1280)

    # Determine type(GPU or not)
    if torch.cuda.is_available():
        net.to(device=torch.device("cuda"))
        net_type = torch.cuda.FloatTensor
    else:
        net_type = torch.FloatTensor

    # Initlialization
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    if cfg.regression:
        criterion = MSELoss()
    else:
        criterion = CrossEntropyLoss()

    # Print net info and log parameters
    print_training_info(net, reds_size)
    log_mlflow_param(cfg)

    # Training loop
    if cfg.small_dataset:
        run_train_small(cfg, ckp_path, save_path, net, net_type, optimizer, criterion)
    else:
        run_train(cfg, ckp_path, save_path, net, net_type, criterion)


if __name__ == "__main__":

    args = parse_args()

    run_train(args)
