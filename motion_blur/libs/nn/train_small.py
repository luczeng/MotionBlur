from torch.utils.data import DataLoader
from motion_blur.libs.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint, log_mlflow_param
from motion_blur.libs.data.dataset import Dataset_OneImage
from motion_blur.libs.utils.training_utils import print_info_small_dataset
from motion_blur.libs.metrics.metrics import evaluate_one_image
import mlflow
import mlflow.pytorch


def run_train_small(config, ckp_path, save_path, net, net_type, optimizer, criterion):
    """
        Training loop for a small dataset (ie not many images)
        Loss is displayed at each epoch instead of at each iteration.
        This script is useful eg to evaluate the capacity ot the network.

        :param config
        :param ckp_path path to checkpoint
        :param save_path path to final model
        :param net
        :param net_type cpu or gpu
    """

    # Resume
    if ckp_path.exists():
        start = load_checkpoint(ckp_path, net, optimizer)
    else:
        start = 0

    # Data
    dataset = Dataset_OneImage(config.mini_batch_size, config.train_dataset_path, config.L_min, config.L_max, net_type)
    dataloader = DataLoader(dataset, batch_size=config.mini_batch_size, shuffle=True)

    # Training loop
    iterations = 0
    running_loss = 0.0
    for epoch in range(start, config.n_epoch):
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
            if (epoch % config.loss_period == (config.loss_period - 1)) & (epoch != 0):

                # Mlflow loggin
                mlflow.log_metric("train_loss", running_loss / iterations, step=epoch + idx)

                # Print training info
                running_loss, iterations = print_info_small_dataset(
                    running_loss, iterations, epoch, idx, len(dataset), config
                )
                print("\t\t", x[0, :].cpu().detach().numpy(), batch["gt"][0, :].cpu().numpy())

            # Run evaluation
            if (epoch % config.validation_period == config.validation_period - 1) & epoch != 0:
                angle_loss, length_loss = evaluate_one_image(
                    net, config.val_small_dataset_path, net_type, config.val_n_angles, config.val_n_lengths
                )
                mlflow.log_metric("angle_error", angle_loss.item())
                mlflow.log_metric("length_error", length_loss.item())
                print(
                    f"\t\t Validation set: Angle error: {angle_loss.item():.2f}, Length error: {length_loss.item():.2f}"
                )

            if epoch % config.saving_epoch == config.saving_epoch - 1:
                # Checkpoint, save checkpoint to disck
                ckp = define_checkpoint(net, optimizer, epoch)
                save_checkpoint(ckp, ckp_path)

    # Run evaluation
    angle_loss, length_loss = evaluate_one_image(
        net, config.val_small_dataset_path, net_type, config.val_n_angles, config.val_n_lengths
    )
    mlflow.log_metric("final_angle_error", angle_loss.item())
    mlflow.log_metric("final_length_error", length_loss.item())

    # Upload model in mlflow
    if config.log_weights:
        mlflow.pytorch.log_model(net, "models")
