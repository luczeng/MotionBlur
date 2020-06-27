from torch.utils.data import DataLoader
from motion_blur.libs.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint
from motion_blur.libs.data.dataset_small import DatasetOneImageRegression
from motion_blur.libs.utils.training_utils import print_info_small_dataset
from motion_blur.libs.metrics.metrics_small_dataset import evaluate_one_image_regression
import mlflow
import mlflow.pytorch
import numpy as np


def run_train_small_regression(cfg, ckp_path, save_path, net, net_type, optimizer, criterion):
    """
        Training loop for a small dataset (ie not many images)
        Loss is displayed at each epoch instead of at each iteration.
        This script is useful eg to evaluate the capacity ot the network.

        :param cfg
        :param ckp_path path to checkpoint
        :param save_path path to final model
        :param net
        :param net_type cpu or gpu
    """

    # Resume
    start = 0
    if ckp_path.exists() and cfg.load_checkpoint:
        start = load_checkpoint(ckp_path, net, optimizer)

    # Data
    dataset = DatasetOneImageRegression(
        cfg.mini_batch_size, cfg.train_dataset_path, cfg.L_min, cfg.L_max, net_type, cfg.as_gray
    )
    dataloader = DataLoader(dataset, batch_size=cfg.mini_batch_size, shuffle=True)

    # Training loop
    iterations = 0
    running_loss = 0.0
    for epoch in range(start, cfg.n_epoch):
        for idx, batch in enumerate(dataloader):

            # GPU
            net.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            x = net.forward(batch["image"])

            # Backward pass
            loss = criterion(x, batch["gt"])
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            iterations += 1

            # Print info, logging
            if (epoch % cfg.loss_period == (cfg.loss_period - 1)) & (epoch != 0):

                # Mlflow loggin
                mlflow.log_metric("train_loss", running_loss / iterations, step=epoch + idx)

                # Print training info
                running_loss, iterations = print_info_small_dataset(
                    running_loss, iterations, epoch, idx, len(dataset), cfg
                )
                print(
                    "\t\t 1st sample estimates/gt:",
                    np.argmax(x[0].cpu().detach().numpy()),
                    batch["gt"][0].cpu().numpy(),
                )

            # Run evaluation
            if (epoch % cfg.validation_period == cfg.validation_period - 1) & epoch != 0:
                angle_loss = evaluate_one_image_regression(
                    net, cfg.val_small_dataset_path, net_type, cfg.n_angles, cfg.L_min, cfg.L_max, cfg.as_gray
                )
                mlflow.log_metric("angle_error", angle_loss.item())
                print(f"\t\t Validation set: Angle error: {angle_loss.item():.2f}")

            if epoch % cfg.saving_epoch == cfg.saving_epoch - 1:
                # Checkpoint, save checkpoint to disck
                ckp = define_checkpoint(net, optimizer, epoch)
                save_checkpoint(ckp, ckp_path)

    # Run evaluation
    angle_loss = evaluate_one_image_regression(
        net, cfg.val_small_dataset_path, net_type, cfg.n_angles, cfg.L_min, cfg.L_max, cfg.as_gray
    )
    mlflow.log_metric("final_angle_error", angle_loss[0].item())

    # Upload model in mlflow
    if cfg.log_weights:
        mlflow.pytorch.log_model(net, "models")
