from functions import *
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import numpy as np

I = cv2.imread("lena.jpeg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
L = 5
theta = 60
Lambda =1

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

im = Image(I)
kernel = MotionKernel(theta,L)
blurredIm = im.LinearBlur(50,11,kernel)
UnblurredIm = Wiener(blurredIm.image, kernel, Lambda)

f = plt.figure(figsize=(14,7))
gs = matplotlib.gridspec.GridSpec(2, 2)
print(gs)
ax0 =plt.subplot(gs[0])
ax0.imshow(im.image,cmap='gray')
ax1 =plt.subplot(gs[1])
ax1.imshow(blurredIm.image,cmap='gray')
ax2 =plt.subplot(gs[2])
ax2.imshow(UnblurredIm,cmap='gray')


plt.tight_layout()
plt.show()

