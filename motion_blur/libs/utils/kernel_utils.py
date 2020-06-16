from motion_blur.libs.forward_models.linops.convolution import Convolution
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
import numpy as np


class Rotations:
    """ Helper class to generate rotation kernel at several angles and length"""

    def __init__(self, image, L, NAngles):
        """
            :param image
            :param L length of linear motion blur
            :param NAngles number of angles
        """
        self.image = image
        self.NAngles = NAngles
        self.L = L
        self.Angles = sorted(np.linspace(0, 180, NAngles))

    def Apply(self):
        self.Out = np.zeros((self.image.shape[0], self.image.shape[1], self.NAngles))
        self.Kernels = [None] * self.NAngles

        for i in range(self.NAngles):
            self.Kernels[i] = Convolution(motion_kernel(self.Angles[i], self.L))
            self.Out[:, :, i] = self.Kernels[i] * self.image
