from numpy.fft import fft2, ifft2
import numpy as np
import math

##################################################################################################################################
class Image:
    def __init__(self, image):
        self.image = image
        self.shape = image.shape

    def LinearBlur(self, theta, L, h):
        # Theta : angle with respects to the vertical axis
        # L 	   : Number of pixels of the blur (odd)

        kernel = np.zeros([L, L])
        pos = int((L - 1) / 2)
        kernel[pos, :] = 1

        out = Image(np.real(ifft2(fft2(self.image) * fft2(h, self.shape))))

        return out


##################################################################################################################################
def vector_coord(angle, length):
    """
        TODO
    """
    cartesianAngleRadians = (450 - angle) * math.pi / 180.0
    x = length * math.cos(cartesianAngleRadians)
    y = length * math.sin(cartesianAngleRadians)
    return x, y
