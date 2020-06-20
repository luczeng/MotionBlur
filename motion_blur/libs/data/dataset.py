from skimage import io
from torch.utils.data import Dataset
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import torch
import random
from pathlib import Path


class Dataset(Dataset):
    def __init__(self, root_dir, L_min, L_max, net_type):
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


class Dataset_OneImage(Dataset):
    def __init__(self, batch_size, root_dir, L_min, L_max, net_type, as_gray=True):
        """
            This dataset is being used for evaluating the capacity of the model on one image
            It only uses one image

            TODO: remove list
        """
        self.root_dir = root_dir
        self.L_min = L_min
        self.L_max = L_max
        if L_min != L_max:
            self.length_list = torch.arange(L_min, L_max, 2)  # odd values
        else:
            self.length_list = [torch.tensor([L_min]).float()]
            self.n_lengths = len(self.length_list)
        self.img_list = [img_path for img_path in Path(root_dir).iterdir() if img_path.is_file()]
        self.net_type = net_type
        self.batch_size = batch_size
        self.as_gray = as_gray

    def __getitem__(self, idx):

        L = self.length_list[random.randint(0, self.n_lengths - 1)]
        theta = torch.rand(1) * 180

        img = io.imread(self.img_list[0], as_gray=self.as_gray)

        gt = torch.cat((theta, L)).type(self.net_type)

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
        return self.batch_size
