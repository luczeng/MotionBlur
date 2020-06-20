import numpy as np
from motion_blur.libs.forward_models.linops.convolution import Convolution


def test_grascale_convolution():

    # 1 kernel
    x = np.ones((5, 5))
    kernel = np.ones((1, 1))

    H = Convolution(kernel)
    y = H * x

    np.testing.assert_array_equal(y, x)

    # 2x2 1 kernel
    x = np.ones((5, 5))
    kernel = np.ones((2, 2)) / 4

    H = Convolution(kernel)
    y = H * x

    np.testing.assert_array_equal(y, x)

    # Edge detector in x direction
    x = np.ones((5, 5))
    kernel = np.array([-1, 1])
    kernel = kernel[None, :]

    H = Convolution(kernel)
    y = H * x

    y_gt = np.zeros((5, 5))

    np.testing.assert_array_equal(y, y_gt)

    # Edge detector, more complicated input
    x = np.zeros((5, 5))
    x[:, 2] = 1
    kernel = np.array([-1, 1])
    kernel = kernel[None, :]

    H = Convolution(kernel)
    y = H * x

    y_gt = np.zeros((5, 5))
    y_gt[:, 2] = -1
    y_gt[:, 3] = 1

    np.testing.assert_almost_equal(y, y_gt, decimal=10)


def test_color_convolution():

    # TODO: need more tests here

    # 2x2 1 kernel
    x = np.ones((5, 5, 3))
    kernel = np.ones((2, 2)) / 4

    H = Convolution(kernel)
    y = H * x

    np.testing.assert_array_equal(y, x)
