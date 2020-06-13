from motion_blur.libs.nn.motion_net import MotionNet
from motion_blur.libs.configs.read_config import parse_config
from motion_blur.libs.nn.train import run_train
from motion_blur.libs.nn.train_small import run_train_small
from motion_blur.libs.utils.nn_utils import print_training_info
from pathlib import Path
import argparse
import torch
import torch.optim as optim
from torch.nn import MSELoss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()

    # Configs
    config = parse_config(args.config_path)

    # Path
    Path(config.save_path).mkdir(parents=True, exist_ok=True)
    ckp_path = Path(config.save_path) / "ckp.pth"
    save_path = Path(config.save_path) / "final_model.pth"

    # Net
    reds_size = [1280, 720]
    net = MotionNet(config.n_layers, config.n_sublayers, config.n_features_first_layer, reds_size)

    # Determine type (GPU or not)
    if torch.cuda.is_available():
        net.to(device=torch.device("cuda"))
        net_type = torch.cuda.FloatTensor
    else:
        net_type = torch.FloatTensor

    # Initlialization
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    criterion = MSELoss()

    # Training loop
    print_training_info(net, reds_size)
    if config.small_dataset:
        run_train_small(config, ckp_path, save_path, net, net_type, optimizer, criterion)
    else:
        run_train(config, ckp_path, save_path, net, net_type, criterion)
