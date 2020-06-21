from skimage import io
from torch.utils.data import Dataset
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import torch
import random
from pathlib import Path


class DatasetOneImageRegression(Dataset):
    def __init__(self, batch_size, root_dir, L_min, L_max, net_type, as_gray=True):
        """
            This dataset is being used for evaluating the capacity of the model on one image
            It only uses one image
            The angles are genereted continuously

            :param batch_size
            :param root_dir path to directory containing dataset (should contain only 1 image)
            :param L_min minimum length of the blur
            :param L_max maximum length of the blur
            :param n_angles discretisation of the angle space, ie how many angles are taken in [0, 180]°
            :param net_type type of the network (cuda float, cpu float etc...)
            :param as_gray True is the image should be loaded in grayscale
        """
        self.root_dir = root_dir
        self.L_min = L_min
        self.L_max = L_max

        # List of lengths (discrete)
        if L_min != L_max:
            self.length_list = torch.arange(L_min, L_max, 2).float()  # odd values
        else:
            self.length_list = [torch.tensor(L_min).float()]
        self.n_lengths = len(self.length_list)

        self.net_type = net_type
        self.batch_size = batch_size

        # Prepare image
        self.img_list = [img_path for img_path in Path(root_dir).iterdir() if img_path.is_file()]
        if len(self.img_list) > 1:
            raise ValueError("The folder should contain only one image")
        self.as_gray = as_gray
        self.img = io.imread(self.img_list[0], as_gray=self.as_gray)

    def __getitem__(self, idx):

        L = self.length_list[random.randint(0, self.n_lengths - 1)]
        theta = torch.rand(1) * 180

        img = io.imread(self.img_list[0], as_gray=self.as_gray)

        gt = torch.cat((theta, L.reshape(1))).type(self.net_type)

        kernel = motion_kernel(theta, int(L))
        H = Convolution(kernel)

        if self.as_gray:
            img = torch.tensor((H * self.img)[None, :, :]).type(self.net_type)
        else:
            img = torch.tensor((H * self.img)[:, :]).type(self.net_type)
            img = img.permute(2, 0, 1)

        sample = {"image": img, "gt": gt}

        return sample

    def __len__(self):
        return self.batch_size


class DatasetOneImageClassification(Dataset):
    def __init__(self, batch_size, root_dir, L_min, L_max, n_angles, net_type, as_gray=True):
        """
            This dataset is being used for evaluating the capacity of the model on one image
            The generated angles are discrete
            It only uses one image

            :param batch_size
            :param root_dir path to directory containing dataset (should contain only 1 image)
            :param L_min minimum length of the blur
            :param L_max maximum length of the blur
            :param n_angles discretisation of the angle space, ie how many angles are taken in [0, 180]°
            :param net_type type of the network (cuda float, cpu float etc...)
            :param as_gray True is the image should be loaded in grayscale
        """
        self.root_dir = root_dir
        self.L_min = L_min
        self.L_max = L_max
        self.net_type = net_type
        self.batch_size = batch_size

        # List of lengths (discrete)
        if L_min != L_max:
            self.length_list = torch.arange(L_min, L_max, 2).float()  # odd values
        else:
            self.length_list = [torch.tensor(L_min).float()]
        self.n_lengths = len(self.length_list)

        # List of angles
        self.angle_list = torch.linspace(0, 180, n_angles).float()  # odd values
        self.n_angles = n_angles

        # Prepare image
        self.img_list = [img_path for img_path in Path(root_dir).iterdir() if img_path.is_file()]
        if len(self.img_list) > 1:
            raise ValueError("The folder should contain only one image")
        self.as_gray = as_gray
        self.img = io.imread(self.img_list[0], as_gray=self.as_gray)

    def __getitem__(self, idx):

        idx_L = random.randint(0, self.n_lengths - 1)
        idx_theta = random.randint(0, self.n_angles - 1)
        L = self.length_list[idx_L]
        theta = self.angle_list[idx_theta]

        img = io.imread(self.img_list[0], as_gray=self.as_gray)

        # gt = torch.cat((torch.tensor(idx_theta).reshape(1), torch.tensor(idx_L).reshape(1))).type(torch.cuda.LongTensor)
        gt = torch.tensor(idx_theta).type(torch.cuda.LongTensor)

        kernel = motion_kernel(theta, int(L))
        H = Convolution(kernel)

        if self.as_gray:
            img = torch.tensor((H * self.img)[None, :, :]).type(self.net_type)
        else:
            img = torch.tensor((H * self.img)[:, :]).type(self.net_type)
            img = img.permute(2, 0, 1)

        sample = {"image": img, "gt": gt}

        return sample

    def __len__(self):
        return self.batch_size
