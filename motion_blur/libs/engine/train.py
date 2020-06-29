from torch.utils.data import DataLoader
from motion_blur.libs.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint
from motion_blur.libs.metrics.metrics import run_validation_classification
from motion_blur.libs.data.dataset import DatasetClassification
from motion_blur.libs.utils.training_utils import print_info
import numpy as np
import torch
import mlflow


def run_train_full_classification(cfg, ckp_path, save_path, net, net_type, optimizer, criterion):
    """
        Main training loop

        :param cfg
        :param ckp_path path to checkpoint
        :param save_path path to final model
        :param net
        :param net_type cpu or gpu
        :param optimizer
        :param criterion
    """

    # Resume
    if ckp_path.exists():
        start = load_checkpoint(ckp_path, net, optimizer)
    else:
        start = 0

    # Data
    dataset = DatasetClassification(cfg.train_dataset_path, cfg.L_min, cfg.L_max, cfg.n_angles, net_type, cfg.as_gray)
    dataloader = DataLoader(dataset, batch_size=cfg.mini_batch_size, shuffle=True)

    # Training loop
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

            # Print loss
            if idx % cfg.loss_period == (cfg.loss_period - 1):
                running_loss = print_info(running_loss, epoch, idx, len(dataset), cfg)
                print(
                    "\t\t 1st sample estimates/gt:",
                    np.argmax(x[0].cpu().detach().numpy()),
                    batch["gt"][0].cpu().numpy(),
                )

            # Run validation
            if cfg.use_validation & (idx % cfg.validation_period == cfg.validation_period - 1):
                angle_error, length_error = run_validation_classification(cfg, net, net_type)
                print(f"\t\tValidation error: angle = {angle_error}, length = {length_error}")

            if epoch % cfg.saving_epoch == cfg.saving_epoch - 1:
                # Checkpoint, save checkpoint to disck
                ckp = define_checkpoint(net, optimizer, epoch)
                save_checkpoint(ckp, ckp_path)

    torch.save(net.state_dict(), save_path)

    # Run evaluation
    angle_loss = run_validation_classification(
        net, cfg.val_small_dataset_path, net_type, cfg.n_angles, cfg.L_min, cfg.L_max, cfg.as_gray
    )
    mlflow.log_metric("final_angle_error", angle_loss.item())

    # Upload model in mlflow
    if cfg.log_weights:
        mlflow.pytorch.log_model(net, "models")
