import torch
from pathlib import Path
from skimage import io
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import random


def create_validation_dataset_regression(cfg):
    """
        Blurs and saves images for the validation dataset. Fill in parameters in your cfg file

        :parameter cfg
        TODO write test for this
    """

    Path(cfg.blurred_val_dataset_path).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)  # nota bene: the seed does not guarantee same numbers on all machines

    for img_path in Path(cfg.sharp_val_dataset_path).iterdir():

        # Blur image
        L = cfg.L_min + torch.rand(1) * (cfg.L_max - cfg.L_min)
        L = L.round()
        theta = torch.rand(1) * 180

        img = io.imread(img_path, as_gray=True)

        gt = torch.cat((theta, L))

        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)
        img = torch.tensor((H * img)[None, None, :, :])
        img = (img * 255 / img.max()).type(torch.uint8)  # to have an image between 0 - 255

        # Save
        outpath = Path(cfg.blurred_val_dataset_path) / (img_path.stem + ".png")
        gt_path = Path(cfg.blurred_val_dataset_path) / (img_path.stem + "_gt.pt")

        io.imsave(outpath, img.numpy()[0, 0, :, :])
        torch.save(gt, gt_path)


def create_validation_dataset_classification(cfg):
    """
        Blurs and saves images for the validation dataset. Fill in parameters in your config file

        :parameter cfg
        TODO write test for this
    """

    Path(cfg.blurred_val_dataset_path).mkdir(parents=True, exist_ok=True)

    # List of lengths (discrete)
    if cfg.L_min != cfg.L_max:
        length_list = torch.arange(cfg.L_min, cfg.L_max, 2).float()  # odd values
    else:
        length_list = [torch.tensor(cfg.L_min).float()]
    angle_list = torch.linspace(0, 179, cfg.n_angles).float()  # odd values

    torch.manual_seed(0)  # nota bene: the seed does not guarantee same numbers on all machines
    for img_path in Path(cfg.sharp_val_dataset_path).iterdir():

        # Blur image
        idx_L = random.randint(0, cfg.n_lengths - 1)
        idx_theta = random.randint(0, cfg.n_angles - 1)
        L = length_list[idx_L]
        theta = angle_list[idx_theta]

        img = io.imread(img_path, as_gray=True)
        gt = torch.cat((theta.reshape(1), L.reshape(1)))

        kernel = motion_kernel(theta, int(L))
        H = Convolution(kernel)
        img = torch.tensor((H * img)[None, None, :, :])
        img = (img * 255 / img.max()).type(torch.uint8)  # to have an image between 0 - 255

        # Save
        outpath = Path(cfg.blurred_val_dataset_path) / (img_path.stem + ".png")
        gt_path = Path(cfg.blurred_val_dataset_path) / (img_path.stem + "_gt.pt")

        io.imsave(outpath, img.numpy()[0, 0, :, :])
        torch.save(gt, gt_path)
