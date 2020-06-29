from pathlib import Path
from skimage import io
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import torch


def weighted_mse_loss(inferences, ground_truth, weights) -> float:
    """
        Calculates MSE loss with weights for unbalanced class
        :param inferences
        :param ground_truth:
        :param weights
        :return loss
    """
    loss = 0
    for inference, gt in zip(inferences, ground_truth):
        loss += torch.sum(weights * (inference - gt) ** 2)
    return loss


def run_validation_regression(config, net, net_type):
    """
        Calculates the average error on the validation set
        :param config
        :param net
        :param net_type cuda or cpu type
        :return angle_loss, length_loss

        TODO: switch to dataloader to benefit from better accelerations?
    """

    img_path_list = Path(config.blurred_val_dataset_path).glob("**/*.png")

    angle_loss = 0
    length_loss = 0
    n_samples = 0

    for path in img_path_list:
        gt_path = Path(config.blurred_val_dataset_path) / (path.stem + "_gt.pt")

        # Load data
        img = io.imread(path)
        gt = torch.load(gt_path)
        img = torch.tensor(img).type(net_type)
        img /= img.max()
        img = img[None, None, :, :]

        # Infer
        x = net.forward(img)

        angle_loss += torch.abs(x[:, 0] - gt[0]).detach()
        length_loss += torch.abs(x[:, 1] - gt[1]).detach()
        n_samples += 1

    angle_loss /= n_samples
    length_loss /= n_samples

    return angle_loss, length_loss


def run_validation_classification(config, net, net_type):
    """
        Calculates the average error on the validation set
        :param config
        :param net
        :param net_type cuda or cpu type
        :return angle_loss, length_loss

        TODO: switch to dataloader to benefit from better accelerations?
    """

    img_path_list = Path(config.blurred_val_dataset_path).glob("**/*.png")

    angle_loss = 0
    length_loss = 0
    n_samples = 0

    for path in img_path_list:
        gt_path = Path(config.blurred_val_dataset_path) / (path.stem + "_gt.pt")

        # Load data
        img = io.imread(path)
        gt = torch.load(gt_path)
        img = torch.tensor(img).type(net_type)
        img /= img.max()
        img = img[None, None, :, :]

        # Infer
        x = net.forward(img)

        angle_loss += torch.abs(x[:, 0] - gt[0]).detach()
        length_loss += torch.abs(x[:, 1] - gt[1]).detach()
        n_samples += 1

    angle_loss /= n_samples
    length_loss /= n_samples

    return angle_loss, length_loss
