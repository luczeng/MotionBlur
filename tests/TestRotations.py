##################################################################################################################################################################
##################################################################################################################################################################
#Show the result of applying the motion blur with several angles
#
# NAngles 	:		number of angles, uniformly generated between 0 and 180°
# L 		:		length of the blur
##################################################################################################################################################################
##################################################################################################################################################################
import numpy as np
import cv2, time
from IPython import display
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './functions')
from functions import *

NAngles = 20
L = 15

##################################################################################################################################################################
#Generate blur
In = cv2.imread("lena.jpeg",0)
RotatedIm = Rotations(In,L,NAngles)
RotatedIm.Apply()

##################################################################################################################################################################
#Display
fig= plt.figure(figsize=(11.5,7.5))
gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
ax_slider = fig.add_axes([0.1, 0.05, 0.8, 0.04])
slider = PageSlider(ax_slider, 'Page', NAngles, activecolor="orange")
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(RotatedIm.Out[:,:,0],cmap='gray')
ax2 = plt.subplot(gs[1])
im2 = ax2.imshow(RotatedIm.Kernels[:,:,0],cmap='gray')

def update(val):
	i = int(slider.val)
	im1.set_data(RotatedIm.Out[:,:,i])
	im2.set_data(RotatedIm.Kernels[:,:,i])

slider.on_changed(update)

plt.show() 
