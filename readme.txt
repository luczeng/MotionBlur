#A small project to estimate the moving blur of a camera using Deep Learning.

There are two parts to this project : 
	- the Image Processing Part with the deconvolution algorithms
	- the Deep Learning part that aims to estimate the blur

The blur is for the moment linear. Currently, we only try to estimate the angle, later we'll switch to the length.

#The project is coded using Keras with a Tensorflow backend
#The libraries needed are :
	- IPython
	- matplotlib
	- sklearn
	- opencv (installed as cv2)
	- numpy
	- scipy

#The linear kernel is obtained by integrating a line over one pixel