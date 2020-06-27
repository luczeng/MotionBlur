import matplotlib.pyplot as plt
import cv2
from motion_blur.libs.forward_models.functions import Image
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.inverse_problems.wiener import Wiener
from motion_blur.libs.forward_models.linops.convolution import Convolution
from matplotlib.gridspec import GridSpec


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


if __name__ == "__main__":
    """
        Shows result of Wiener devoncolution of a linear motion blur
    """

    # Parameters
    L = 23
    theta = 30
    Lambda = 0.001

    # Load image, blur and deblur
    reds_test_img = "datasets/reds_small/000_00000000.png"
    lena = "imgs/lena.tiff"
    img = cv2.imread(reds_test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = motion_kernel(theta, L)
    H = Convolution(kernel)
    img_blur = H * img
    # half_size =int((L-1)/2)
    # img_blur = img_blur[half_size:-half_size,half_size:-half_size]
    UnblurredIm = Wiener(img_blur, kernel, Lambda)

    # Display
    f = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2)
    ax0 = plt.subplot(gs[0])
    ax0.imshow(img, cmap="gray")
    ax0.set_title("Original image")
    ax0.axis("off")
    ax1 = plt.subplot(gs[1])
    ax1.imshow(img_blur, cmap="gray")
    ax1.set_title("Blurry image")
    ax1.axis("off")
    ax2 = plt.subplot(gs[2])
    ax2.imshow(UnblurredIm, cmap="gray")
    ax2.set_title("Restored image")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()
