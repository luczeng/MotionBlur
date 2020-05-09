# Description
This project aims at removing motion blur originating from the motion or shake of hand-held cameras. It aims to work blindly, ie no knowledge of the blur is required. The motion blur is estimated using a convolutional neural network, and is later used to calibrate a deconvolution algorithm.  

The project consists of two distinct parts:  
	- the image processing section, with the deconvolution algorithms and the forward models.  
	- the blur estimation section using a neural network.  
	
See the [wiki](https://github.com/luczeng/MotionBlur/wiki) for some visual insights.  

The library is coded in Python3.

# News
- As of May 2020, the project restarts! We move from tensorflow to pytorch. We will extend the motion blur models to space variant kernels. We plan to extend to TV deblurring. We will try better backbones (resnets).

# Progress
- As of now May 2020, we support deblurring of *linear blur* using a Wiener filter

# Installation
In your favorite conda environment, type:
~~~
    pip install -e .
~~~

For development, install the test libraries as follow:

~~~
    pip install -e .[TEST_SUITE]
~~~

# Implementation
- The linear kernel is obtained by integrating a line over one pixel so as to take into account discretization effects.
- The deconvolution is a Wiener filter. We plan to add a TV deconvolution.
- Data and training:  WIP

# Usage
- WIP

# Performance
<p  align="center">
	<img src="imgs/results.jpg" width="400" alt="Results">
	<br>
	<em> Estimated angles for a linear motion blur </em>
</p>

