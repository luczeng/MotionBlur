import numpy as np
import cv2, time
from functions import *
from IPython import display
import matplotlib.pyplot as plt

In = cv2.imread("lena.jpeg",0)
NAngles = 20

RotatedIm = Rotations(In,19,NAngles)
RotatedIm.Apply()

print(RotatedIm.Angles)
'''
plt.ion()
#f, ax = plt.subplots()


for i in range(RotatedIm.NAngles):
		plt.imshow(RotatedIm.Out[i])
		plt.pause(0.1)

while True:
	plt.pause(0.05)
'''

fig= plt.figure(figsize=(10,6.5))
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
