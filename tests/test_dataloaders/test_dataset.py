from motion_blur.libs.data.dataset_small import DatasetOneImageRegression
from motion_blur.libs.data.dataset_small import DatasetOneImageClassification
import unittest
import torch
import numpy as np


class TestDatasetOneImageRegression(unittest.TestCase):
    """
        Tests the regression dataset options
    """

    def test_dataset_1_image_1_length_gray(test):

        dataset = DatasetOneImageRegression(2, "tests/test_data/test_img", 3, 3, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_1_length_rgb(test):

        dataset = DatasetOneImageRegression(2, "tests/test_data/test_img", 3, 3, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_different_lengths_gray(test):

        dataset = DatasetOneImageRegression(2, "tests/test_data/test_img", 3, 5, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_different_lengths_rgb(test):

        dataset = DatasetOneImageRegression(2, "tests/test_data/test_img", 3, 5, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        assert x["gt"].shape == torch.Size([2])


class TestDatasetOneImageClassification(unittest.TestCase):
    """
        Tests the classification dataset options
    """

    def test_dataset_1_image_1_length_gray(test):

        dataset = DatasetOneImageClassification(2, "tests/test_data/test_img", 3, 3, 4, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        np.testing.assert_equal(dataset.angle_list.numpy(), np.array([0, 60, 120, 180]))

    def test_dataset_1_image_1_length_rgb(test):

        dataset = DatasetOneImageClassification(2, "tests/test_data/test_img", 3, 3, 4, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        np.testing.assert_equal(dataset.angle_list.numpy(), np.array([0, 60, 120, 180]))

    def test_dataset_1_image_different_lengths_gray(test):

        dataset = DatasetOneImageClassification(2, "tests/test_data/test_img", 3, 5, 4, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        np.testing.assert_equal(dataset.angle_list.numpy(), np.array([0, 60, 120, 180]))

    def test_dataset_1_image_different_lengths_rgb(test):

        dataset = DatasetOneImageClassification(2, "tests/test_data/test_img", 3, 5, 4, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        np.testing.assert_equal(dataset.angle_list.numpy(), np.array([0, 60, 120, 180]))
