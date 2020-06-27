import cv2
import matplotlib.pyplot as plt
from motion_blur.libs.utils.kernel_utils import Rotations
from motion_blur.libs.utils.display_utils import PageSlider
from matplotlib.gridspec import GridSpec


"""
    Interactively displays the result of the convolution with several motion kernels
"""
# Parameters
NAngles = 10
L = 41

# Generate blur
img = cv2.imread("imgs/lena.tiff", 0)
RotatedIm = Rotations(img, L, NAngles)
RotatedIm.Apply()

# Display
fig = plt.figure(figsize=(11.5, 7.5))
gs = GridSpec(1, 2, width_ratios=[6, 1])
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
slider = PageSlider(ax_slider, "Page", NAngles, activecolor="orange")
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(RotatedIm.Out[:, :, 0], cmap="gray")
ax2 = plt.subplot(gs[1])
im2 = ax2.imshow(RotatedIm.Kernels[0].kernel, cmap="gray")


def update(val):
    """ Updates the axis with the new kernel and blurred image """
    i = int(slider.val)
    im1.set_data(RotatedIm.Out[:, :, i])
    im2.set_data(RotatedIm.Kernels[i].kernel)


slider.on_changed(update)
plt.show()
