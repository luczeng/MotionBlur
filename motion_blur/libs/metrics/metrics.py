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


def create_validation_dataset(config):
    """
        Blurs and saves images for the validation dataset. Fill in parameters in your config file

        :parameter config
    """

    Path(config.blurred_val_dataset_path).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)  # nota bene: the seed does not guarantee same numbers on all machines

    for img_path in Path(config.sharp_val_dataset_path).iterdir():

        # Blur image
        L = config.L_min + torch.rand(1) * (config.L_max - config.L_min)
        L = L.round()
        theta = torch.rand(1) * 180

        img = io.imread(img_path, as_gray=True)

        gt = torch.cat((theta, L))

        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)
        img = torch.tensor((H * img)[None, None, :, :])
        img = (img * 255 / img.max()).type(torch.uint8)  # to have an image between 0 - 255

        # Save
        outpath = Path(config.blurred_val_dataset_path) / (img_path.stem + ".png")
        gt_path = Path(config.blurred_val_dataset_path) / (img_path.stem + "_gt.pt")

        io.imsave(outpath, img.numpy()[0, 0, :, :])
        torch.save(gt, gt_path)


def run_validation(config, net, net_type):
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
