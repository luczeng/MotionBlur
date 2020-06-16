from motion_blur.libs.forward_models.kernels.motion import line_integral, motion_kernel
import unittest
import pytest
import math
import numpy as np


assertions = unittest.TestCase("__init__")


class TestMotionKernel(unittest.TestCase):
    """
        Test the motion kernel generation in the horizontal and vertical directions
        For both the odd and even cases
    """

    def test_kernel_1(test):

        # Kernel parameters
        L = 1
        theta = 35

        kernel = motion_kernel(theta, L)

        # Gt
        kernel_gt = np.array([1])
        kernel_gt = kernel_gt.reshape(1, 1)

        np.testing.assert_array_equal(kernel_gt, kernel)

    def test_kernel_horizontal_11(test):
        # Kernel parameters
        L = 11
        theta = 0

        kernel = motion_kernel(theta, L)

        # Gt
        kernel_gt = np.zeros((int(L), int(L)))
        kernel_gt[5, :] = 1
        kernel_gt /= 11

        np.testing.assert_array_equal(kernel_gt, kernel)

    def test_kernel_vertical_11(test):
        # Kernel parameters
        L = 11
        theta = 90

        kernel = motion_kernel(theta, L)

        # Gt
        kernel_gt = np.zeros((int(L), int(L)))
        kernel_gt[:, 5] = 1
        kernel_gt /= 11

        np.testing.assert_array_equal(kernel_gt, kernel)


class TestLineIntegral(unittest.TestCase):
    """
        Tests the integration of a line over one pixel
        x,y are coordinates of the center of the considered pixel
    """

    def test_line_integral_45_degrees(test):

        # Diagonal case
        theta = 45
        x = 10
        y = 10

        L = line_integral(theta, x, y)

        assertions.assertAlmostEqual(L, math.sqrt(2), 3)

        # Diagonal negative case
        theta = 45
        x = 12
        y = 10

        L = line_integral(theta, x, y)

    def test_line_integral_vertical(self):
        # Vertical case
        theta = 90
        x = 0
        y = 10
        L = line_integral(theta, x, y)

        assert L == 1

        # Vertical negative case
        theta = 90
        x = 20
        y = 10
        L = line_integral(theta, x, y)

        assert L == 0

    def test_line_integral_horizontal(self):
        # Horizontal case
        theta = 0
        x = 10
        y = 0
        L = line_integral(theta, x, y)

        assert L == 1

        # Horizontal negative case
        theta = 0
        x = 10
        y = 30
        L = line_integral(theta, x, y)

        assert L == 0

    def test_line_integral_upper_diagonal(self):
        # To understand this, better draw on paper the line that meets passes at point 1; 1.5
        theta = math.atan(1.5) * 180 / math.pi
        x = 1.5
        y = 1.5
        L_true = math.sqrt(0.5 ** 2 + (1 / 3) ** 2)

        L = line_integral(theta, x, y)

        assertions.assertAlmostEqual(L, L_true, 4)

    def test_line_integral_lower_diagonal(self):
        # To understand this, better draw on paper the line that meets passes at point 1; 1.5
        theta = math.atan(1 / 1.5) * 180 / math.pi
        x = 1.5
        y = 1.5
        L_true = math.sqrt(0.5 ** 2 + (1 / 3) ** 2)

        L = line_integral(theta, x, y)

        assertions.assertAlmostEqual(L, L_true, 4)
