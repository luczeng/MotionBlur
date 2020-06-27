[![Actions Status](https://github.com/luczeng/MotionBlur/workflows/Python%20package/badge.svg)](https://github.com/luczeng/MotionBlur/actions)

# Description
This project aims at removing motion blur originating from the motion or shake of hand-held cameras. It aims to work blindly, ie no knowledge of the blur is required. The motion blur is estimated using a convolutional neural network, and is later used to calibrate a deconvolution algorithm.  

The project consists of two distinct parts:    
	- the image processing section, with the deconvolution algorithms and the forward models.  
	- the blur estimation section using a neural network.  

See the **[wiki](https://github.com/luczeng/MotionBlur/wiki)** for some visual insights.  

The library is coded in Python3.

Contributions are more than welcome, either on on the image processing (modeling of complex blurs) or the blur estimation.

![alt text](imgs/Reds_deconv.png)

# News
- As of May 2020, the project restarts! We move from tensorflow to pytorch. We will extend the motion blur models to more complicated motions than simply linear movements. We will also tackle the space variant case. We plan to extend to TV deblurring. 

# Progress
- As of now (May 2020), we support deblurring of *linear blurs* using a Wiener filter.

# Installation
In your favorite conda environment, type:  
  
~~~
    pip install -e .
~~~

For development, install the test libraries as follow:

~~~
    pip install -e ".[TEST_SUITE,DEVELOP]"
~~~

# Content details
## Forward model
- The linear kernel is obtained by integrating a line over one pixel so as to take into account discretization effects.
- The deconvolution is a Wiener filter. We plan to add a TV deconvolution.

## Learning
- Data and training:  WIP
- Training data is randomly generated on the fly using the forward model. The validation set is generated offline.
- Training is online gradient descent

## Implementation details
- We use Torch and python3. For managing learning runs, we use mlflow with Neptune.

# Usage
- For inference, edit or copy the configuration file in libs/configs and run:  

 
```
    python driver_scripts/main_inference.py -i path_to_config.yml
```
- Training: TBD
```
    python driver_scripts/main_train.py -i path_to_config.yml
```

Nota bene: I plan to upload the weights soon.

# Datasets

We currently use the REDS (or GOPRO) dataset for training. If you know any dataset consisting of sharp images, please let
me know!

# Contributing
I use Black with line length 120. Please write unit tests (pytest) for your code. Please use the
[git-flow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) development
process.

# Performance
- Visual performance of the linear motion blur regression (latest results):

|![alt text](imgs/linear_motion_blur_perfomance_theta.png)             |![alt text](imgs/linear_motion_blur_perfomance_length.png) 
| ------------- |:-------------:|
