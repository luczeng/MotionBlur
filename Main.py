from functions import *
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import numpy as np

I = cv2.imread("lena.jpeg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
L = 10
theta = 60
Lambda =1

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

im = Image(I)
kernel = MotionBlur(theta,L)
blurredIm = im.LinearBlur(50,9,kernel)
UnblurredIm = Wiener(blurredIm.image, kernel, Lambda)
print(type(im))
print(UnblurredIm)

f, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(im.image,aspect = 'auto',cmap='gray')
ax2.imshow(blurredIm.image,aspect = 'auto',cmap='gray')
ax3.imshow(UnblurredIm,aspect = 'auto',cmap='gray')

forceAspect(ax1,aspect=1)
forceAspect(ax2,aspect=1)
forceAspect(ax3,aspect=1)

plt.tight_layout()
plt.show()
mng = plt.get_current_fig_manager()
mng.frame.Maximize(True)
