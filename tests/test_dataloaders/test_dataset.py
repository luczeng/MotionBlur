from motion_blur.libs.data.dataset import Dataset_OneImage
import unittest
import torch


class TestDatasetOneImage(unittest.TestCase):
    def test_dataset_1_image_1_length_gray(test):

        dataset = Dataset_OneImage(2, "tests/test_data/", 3, 3, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_1_length_rgb(test):

        dataset = Dataset_OneImage(2, "tests/test_data/", 3, 3, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_different_lengths_gray(test):

        dataset = Dataset_OneImage(2, "tests/test_data/", 3, 5, torch.float, as_gray=True)
        x = dataset[0]

        assert x["image"].shape == (1, 168, 300)
        assert x["gt"].shape == torch.Size([2])

    def test_dataset_1_image_different_lengths_rgb(test):

        dataset = Dataset_OneImage(2, "tests/test_data/", 3, 5, torch.float, as_gray=False)
        x = dataset[0]

        assert x["image"].shape == (3, 168, 300)
        assert x["gt"].shape == torch.Size([2])
