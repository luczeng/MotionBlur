from numpy.fft import fft2, ifft2
import numpy as np
import math


def Wiener(In, Kernel, Lambda):
    w = np.conj(fft2(Kernel, In.shape)) / (np.conj(fft2(Kernel, In.shape)) * fft2(Kernel, In.shape) + Lambda)
    out = ifft2(w * fft2(In))
    return np.real(out)
