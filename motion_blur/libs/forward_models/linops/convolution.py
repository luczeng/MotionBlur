from numpy.fft import fft2, ifft2
import numpy as np
from motion_blur.libs.base.base_linop import linop


class Convolution(linop):
    def __init__(self, kernel: np.ndarray):
        """
            Linop that performs convolution by overloading the * operator
            The convolution is done in the Fourier domain

            :param kernel kernel to convolve with
        """
        self.kernel = kernel

    def fourier_convolution(self, img: np.ndarray):
        # TODO: Put some checks
        return np.real(ifft2(fft2(img) * fft2(self.kernel, img.shape)))

    def __mul__(self, img: np.ndarray) -> np.ndarray:
        """
            Performs the convolution of the input image with a kernel using Fourier transforms.
            Zero pads the convolution kernel

            :param img colored or grayscale image to be convolved
            :return result
        """

        if img.ndim == 2:
            return self.fourier_convolution(img)
        else:
            if img.shape[2] == 3:
                out_img = np.empty(img.shape)
                out_img[:, :, 0] = self.fourier_convolution(img[:, :, 0])
                out_img[:, :, 1] = self.fourier_convolution(img[:, :, 1])
                out_img[:, :, 2] = self.fourier_convolution(img[:, :, 2])
                return out_img
            else:
                raise ValueError("Incorrect dimension of input image")
