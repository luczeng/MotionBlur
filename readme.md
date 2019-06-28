# Intro 
A small project to estimate the motion blur of a camera using a convolutional neural network

There are two parts to this project : 
	- the Image Processing Part with the deconvolution algorithms
	- the Deep Learning part that aims to estimate the blur

The blur is for the moment linear. Currently, we only try to estimate the angle, later we'll switch to the length.

# Requirements
The project is coded using Keras with a Tensorflow backend
You can install the dependencies by doing:
	- IPython
	- matplotlib
	- sklearn
	- opencv (installed as cv2)
	- numpy
	- scipy

# Implementation
- The linear kernel is obtained by integrating a line over one pixel so as to take into account discretization effects.
- The deconvolution is a Wiener filter. We plan to add a TV deconvolution.
- Data and training: this project has been runned on my personal computer that does not possess a GPU and has limited RAM. Therefore the training data consisted of one image linearly blurred at random angles (but constant blur intensity). This of course yields less than optimal performances on real life data but provides notheless a proof of concept. 

# Usage
Run 


