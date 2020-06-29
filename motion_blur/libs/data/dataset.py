from skimage import io
from torch.utils.data import Dataset
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import torch
from pathlib import Path
import random


class DatasetClassification(Dataset):
    def __init__(self, root_dir, L_min, L_max, n_angles, net_type, as_gray=True):
        self.root_dir = root_dir
        self.L_min = L_min
        self.L_max = L_max
        self.img_list = [img_path for img_path in Path(root_dir).iterdir() if img_path.is_file()]
        self.net_type = net_type

        # List of lengths (discrete)
        if L_min != L_max:
            self.length_list = torch.arange(L_min, L_max, 2).float()  # odd values
        else:
            self.length_list = [torch.tensor(L_min).float()]
        self.n_lengths = len(self.length_list)
        self.angle_list = torch.linspace(0, 179, n_angles).float()  # odd values
        self.n_angles = n_angles
        self.as_gray = as_gray

    def __getitem__(self, idx):

        img = io.imread(self.img_list[idx], as_gray=self.as_gray)

        # Blur parameters
        idx_L = random.randint(0, self.n_lengths - 1)
        idx_theta = random.randint(0, self.n_angles - 1)
        L = self.length_list[idx_L]
        theta = self.angle_list[idx_theta]

        # gt = torch.cat((theta.reshape(1), L.reshape(1))).type(self.net_type)
        gt = torch.tensor(idx_theta).type(torch.cuda.LongTensor)

        # Blur image
        kernel = motion_kernel(theta, int(L))
        H = Convolution(kernel)
        if self.as_gray:
            img = torch.tensor((H * img)[None, :, :]).type(self.net_type)
        else:
            img = torch.tensor((H * img)[:, :]).type(self.net_type)
            img = img.permute(2, 0, 1)

        sample = {"image": img, "gt": gt}

        return sample

    def __len__(self):
        return len(self.img_list)


class DatasetRegression(Dataset):
    def __init__(self, root_dir, L_min, L_max, n_angles, net_type):
        self.root_dir = root_dir
        self.L_min = L_min
        self.L_max = L_max
        self.img_list = [img_path for img_path in Path(root_dir).iterdir() if img_path.is_file()]
        self.net_type = net_type

    def __getitem__(self, idx):

        L = self.L_min + torch.rand(1) * (self.L_max - self.L_min)
        L = L.round()
        theta = torch.rand(1) * 180

        img = io.imread(self.img_list[idx], as_gray=True)

        gt = torch.cat((theta, L)).type(self.net_type)

        kernel = motion_kernel(theta, L)
        H = Convolution(kernel)
        img = torch.tensor((H * img)[None, :, :]).type(self.net_type)

        sample = {"image": img, "gt": gt}

        return sample

    def __len__(self):
        return len(self.img_list)
