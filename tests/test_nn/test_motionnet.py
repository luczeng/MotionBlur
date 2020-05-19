import torch
from motion_blur.libs.nn.motion_net import MotionNet


def test_forwardpass():

    img = torch.rand((1, 1, 512, 512))

    mnet = MotionNet()
    x = mnet.forward([img])

    assert x[0].shape == (1, 2)
