import cv2
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs the motion blur estimation network on a blurred image with a randomly generated kernel"
    )
    parser.add_argument("-i", "--input_path", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    """
        Interactively displays the result of the convolution with several motion kernels
    """

    args = parse_args()
    # Parameters
    theta = 40
    L = 15

    # Generate blur
    img = cv2.imread(args.input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = motion_kernel(theta, L)
    H = Convolution(kernel)
    img_blur = H * img
    img_blur /= img.max()

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(img_blur)
    ax1.axis("off")
    ax2.axis("off")
    plt.show()
