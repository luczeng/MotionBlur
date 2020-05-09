from numpy.fft import fft2, ifft2
import numpy as np
from motion_blur.libs.base.base_linop import linop


class Convolution(linop):
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def __mul__(self, img: np.ndarray) -> np.ndarray:
        """
            Performs the convolution of the input image with a kernel using Fourier transforms.
            Zero pads the convolution kernel
        """

        # TODO: Put some checks
        return np.real(ifft2(fft2(img) * fft2(self.kernel, img.shape)))
