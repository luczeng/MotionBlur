##################################################################################################################################################################
##################################################################################################################################################################
#Predicts the angle of a linear motion blur
#
#Usage :
#	-l (--load_model) : 1 or 0, loads pre-trained model
#	-s (--save_model) : 1 or 0, saves model
#	-p (--path)		  : path to the model to be loaded or saved
#
#Luc Zeng
##################################################################################################################################################################
##################################################################################################################################################################
import numpy as np
#from IPython import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import cv2,sys
sys.path.append('utils')
from Cnn import MovingBlurCnn
from utils.functions import *
from keras.models import Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from config import get_config

##################################################################################################################################################################
##################################################################################################################################################################
#PARAMETERS
NAngles = 100
L = 15
nb_epoch = 2000

##################################################################################################################################################################
##################################################################################################################################################################
#Argument parser
cfg = get_config(sys.argv)

#if (cfg.["save_model"] == 1 and cfg.["path"] == "empty") or (cfg.["load_model"] == 1 and cfg.["path"] == "empty"):
#	error("\nSUPPLY WEIGHT PATH\n")

##################################################################################################################################################################
#Generate blur
print("reached")
In = cv2.imread(cfg.img_path,0)
print(In.shape)
RotatedIm = Rotations(In,L,NAngles)
RotatedIm.Apply()

data = RotatedIm.Out

#Reshape (add empty channel)
data = data[:,:,np.newaxis,:] #Here we should probably normalize the data
data = np.rollaxis(data,3)
angles = np.array(RotatedIm.Angles)
angles = angles[:,np.newaxis]

#Separate data into training and testing
(X_train,X_test,y_train,y_test) = train_test_split(data,angles,test_size = 0.33)


##################################################################################################################################################################
#Define model and train
print("[Info] Training size :{}\nTraining label size {}\n  ".format(X_train.shape,len(y_train)))

if cfg.load_model == 0:
	model = MovingBlurCnn.build(X_train.shape[1], X_train.shape[2], X_train.shape[3])
	model.compile(loss = 'mean_absolute_error',optimizer = 'adam')
	model.fit(X_train,y_train,nb_epoch = nb_epoch,batch_size = 6)
	if cfg.save_model == 1:
		model_json = model.to_json()
		with open("models/" + cfg.path+".json","w") as json_file:
			json_file.write(model_json)
		model.save_weights("models/" + cfg.path+".h5", overwrite=True)
 
else:
	json_file = open(cfg.path+".json",'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights(cfg.path+".h5")

print(model.summary())


##################################################################################################################################################################
# Show results
prediction = model.predict(X_test) 
print(np.c_[prediction,y_test],"\n")
print("MSE :{} \n",np.sqrt(mse(prediction,y_test)))

k = sorted(range(len(prediction)), key=lambda k: prediction[k])

fig = plt.figure(figsize=(12,8.5))
gs = matplotlib.gridspec.GridSpec(1,2,width_ratios = [1,1])
ax1 = plt.subplot(gs[0])
ax1.scatter(np.arange(0,len(k)),y_test[k],c ='b')
ax1.scatter(np.arange(0,len(k)),prediction[k],c ='r')
ax1.legend(("Ground truth","Predictions"))
ax1.set_ylabel("Angles (in degrees)")
ax_list = fig.axes

ax2 = plt.subplot(gs[1])
ax2.imshow(X_test[0,:,:,0])
ax2.set_title("Blurry image and estimated angle")
[x,y] = vector_coord(y_test[0],45)
ax2.quiver(30,30,x,y,color ='b')
[x,y] = vector_coord(prediction[0],45)
ax2.quiver(60,30,x,y,color = 'r')
ax2.legend(("True direction","Estimated direction"))

print(y_test[0])
plt.show()





