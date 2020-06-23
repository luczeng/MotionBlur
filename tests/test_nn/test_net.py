from motion_blur.libs.nn.motion_net import MotionNet
import torch


def test_size_computation():
    """
        Tests the computation of the size at the end of the convolutional layers
    """

    net = MotionNet(2, 2, 8, [128, 128], as_gray=True, regression=True)
    output_shape = net._compute_conv_size()
    assert [29, 29] == output_shape

    net = MotionNet(3, 2, 8, [128, 128], as_gray=True, regression=True)
    output_shape = net._compute_conv_size()
    assert [12, 12] == output_shape


def test_one_pass_regression():
    """
        Test the forward pass with different sizes of the net
    """

    net = MotionNet(4, 2, 8, [256, 256], as_gray=True, regression=True)
    x = torch.rand(4, 1, 256, 256)
    inference = net._one_pass(x)

    assert inference.shape == torch.Size([4, 2])

    net = MotionNet(5, 4, 16, [512, 512], as_gray=True, regression=True)
    x = torch.rand(4, 1, 512, 512)
    inference = net._one_pass(x)

    assert inference.shape == torch.Size([4, 2])


def test_one_pass_classification():
    """
        Test the forward pass with different sizes of the net
    """

    net = MotionNet(4, 2, 8, [256, 256], as_gray=True, n_angles=5, n_lengths=1)
    x = torch.rand(4, 1, 256, 256)
    inference = net._one_pass(x)

    assert inference.shape == torch.Size([4, 5])

    net = MotionNet(5, 4, 16, [512, 512], as_gray=True, n_angles=5, n_lengths=1)

    x = torch.rand(4, 1, 512, 512)
    inference = net._one_pass(x)

    assert inference.shape == torch.Size([4, 5])
