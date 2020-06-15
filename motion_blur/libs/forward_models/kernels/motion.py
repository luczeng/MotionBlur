import numpy as np
import math
import matplotlib.pyplot as plt
from motion_blur.libs.utils.display_utils import Formatter
import torch


def motion_kernel(theta: float, L: int) -> np.ndarray:
    """
        Generates a linear motion blur kernel.
        Only accepts ODD length. This is due to the fact that the even sized kernel brought a lot of issues. Eg,
        how do you define the kernel at angle 0? It should be continuous for theta -> 0 otherwise it would bring issues
        for the learning part. It is simpler to just discard that case. Any ideas regarding this is more than welcome.

        The kernel is a 1x1 matrix of value 1 if the length is under 2.

        :param theta angle in degrees of the kernel
        :param length of the kernel. Odd size.
        :return kernel: motion kernel

        TODO: find a way to add the case of non integer lengths
    """

    # Checks

    if L % 2 == 0:
        raise ValueError("L should be odd")

    if theta > 180 or theta < 0:
        raise ValueError("theta should be between 0 and 180")

    if torch.is_tensor(L):
        L = L.numpy()
    if torch.is_tensor(theta):
        theta = theta.numpy()

    # Kernel computation
    if L >= 2:
        kernel = np.zeros([L, L])
        x = np.arange(0, L, 1) - (L - 1) / 2
        X, Y = np.meshgrid(x, x)

        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if pythagorean_theorem(X[i, j], Y[i, j]) < float(L) / 2:
                    kernel[i, j] = line_integral(theta, X[i, j], -Y[i, j])

        kernel /= kernel.sum()

    else:
        kernel = np.ones([1, 1])

    return kernel


def pythagorean_theorem(corner_x: float, corner_y: float) -> float:
    """
        Applies the pythagorean theorem to calculate the hypothenuse
        :param corner_x length of one corner
        :param corner_x length of the other corner
        return: length of the hypothenuse
    """

    return math.sqrt(corner_x ** 2 + corner_y ** 2)


def line_integral(theta, x, y, pixel_half_width=0.5):
    """
        Computes the line integral over a pixel, that is, the length of the line crossing the pixel
        Uses the pythagorean theorem to compute the kernel.
        TODO: refactor this

        :param theta angle in degrees of the line (between 0 and 360 degrees)
        :param x x-coordinate of the center of the pixel
        :param y y-coordinate of center of the pixel
        :pixel_half_width: size of the pixel
        return: L: line integral
    """
    # Theta : between 0 and 360
    TanTheta = np.tan(np.deg2rad(theta))
    L = 0

    a = x - pixel_half_width
    b = x + pixel_half_width
    alpha = y - pixel_half_width
    beta = y + pixel_half_width

    if theta != 90 and theta != 270 and theta != 0 and theta != 180:  # non vertical or horizontal cases
        if b >= 0 and beta >= 0 and TanTheta > 0:  # upper right quadrant
            if alpha <= TanTheta * a <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = pythagorean_theorem(b - a, TanTheta * b - TanTheta * a)
                else:
                    L = pythagorean_theorem(beta / TanTheta - a, beta - TanTheta * a)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = pythagorean_theorem(b - alpha / TanTheta, TanTheta * b - alpha)
                else:
                    L = pythagorean_theorem(beta / TanTheta - alpha / TanTheta, beta - alpha)

        elif b >= 0 and alpha <= 0 and TanTheta < 0:  # lower right quadrant
            if alpha <= TanTheta * a <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = pythagorean_theorem(b - a, TanTheta * b - TanTheta * a)
                else:
                    L = pythagorean_theorem(alpha / TanTheta - a, alpha - TanTheta * a)
            elif a <= beta / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = pythagorean_theorem(b - beta / TanTheta, TanTheta * b - beta)
                else:
                    L = pythagorean_theorem(alpha / TanTheta - beta / TanTheta, alpha - beta)

        elif a <= 0 and beta >= 0 and TanTheta < 0:  # upper left quadrant
            if alpha <= TanTheta * b <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = pythagorean_theorem(b - a, TanTheta * b - TanTheta * a)
                else:
                    L = pythagorean_theorem(beta / TanTheta - b, beta - TanTheta * b)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = pythagorean_theorem(a - alpha / TanTheta, TanTheta * a - alpha)
                else:
                    L = pythagorean_theorem(beta / TanTheta - alpha / TanTheta, beta - alpha)

        elif a <= 0 and alpha <= 0 and TanTheta > 0:  # lower left quadrant
            if alpha <= TanTheta * b <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = pythagorean_theorem(b - a, TanTheta * b - TanTheta * a)
                else:
                    L = pythagorean_theorem(alpha / TanTheta - b, alpha - TanTheta * b)
            elif a <= beta / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = pythagorean_theorem(a - beta / TanTheta, TanTheta * a - beta)
                else:
                    L = pythagorean_theorem(alpha / TanTheta - beta / TanTheta, alpha - beta)

        elif a < 0 and b > 0 and alpha < 0 and beta > 0:  # case when pixel is at the center
            if alpha <= TanTheta * a <= beta:
                if alpha <= TanTheta * b <= beta:
                    L = pythagorean_theorem(b - a, a * TanTheta - b * TanTheta)
                else:
                    if TanTheta * a < TanTheta * b:
                        L = pythagorean_theorem(beta / TanTheta - a, a * TanTheta - beta)
                    else:
                        L = pythagorean_theorem(alpha / TanTheta - a, a * TanTheta - alpha)
            else:
                if a <= alpha / TanTheta <= b:
                    if alpha <= TanTheta * b <= beta:
                        L = pythagorean_theorem(b - alpha / TanTheta, alpha - b * TanTheta)
                    else:
                        L = pythagorean_theorem(beta / TanTheta - alpha / TanTheta, alpha - beta)
                else:
                    L = pythagorean_theorem(beta / TanTheta - b, TanTheta * b - beta)

    else:
        if theta == 90:  # vertical case
            if (a == 0 and b >= 1) or (a == -1 and b == 0):  # horizontal case
                L = 0.5
            elif a == -0.5 and b >= 0.5:  # horizontal case
                L = 1
        else:
            if (alpha == 0 and beta >= 1) or (alpha == -1 and beta == 0):  # horizontal case
                L = 0.5
            elif alpha == -0.5 and beta >= 0.5:  # horizontal case
                L = 1

    return L


if __name__ == "__main__":
    """
        Script to visualize the linear motion kernel. Change parameters to see its effects.
    """

    # Parameters
    theta = 0
    L = 2

    # Generate kernel
    kernel = motion_kernel(theta, L)

    # Visualize
    x = np.arange(0, L, 1) - (L - 1) / 2

    fig, ax = plt.subplots()
    im = ax.imshow(kernel, interpolation="none", extent=[x[0], x[-1], x[0], x[-1]])

    ax.format_coord = Formatter(im)
    if np.abs(np.tan(np.deg2rad(theta)) * L) <= L + 10:
        ax.plot(x, np.tan(np.deg2rad(theta)) * x, "g")

    plt.show()
