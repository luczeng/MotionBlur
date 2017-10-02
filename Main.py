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
from functions.functions import *
from IPython import display
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from Cnn import MovingBlurCnn
import cv2
from keras.models import Sequential,model_from_json
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
import argparse

##################################################################################################################################################################
##################################################################################################################################################################
#PARAMETERS
NAngles = 40
L = 15
nb_epoch = 512

##################################################################################################################################################################
##################################################################################################################################################################
#Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-s","--save_model",type=int,default = 0,help="save weights or not")
ap.add_argument("-l","--load_model",type=int,default=0,help="load model")
ap.add_argument("-p", "--path", type=str,help="path to model file",default ="empty")
args = vars(ap.parse_args())

if (args["save_model"] == 1 and args["path"] == "empty") or (args["load_model"] == 1 and args["path"] == "empty"):
	error("\nSUPPLY WEIGHT PATH\n")

##################################################################################################################################################################
#Generate blur
In = cv2.imread("lena.jpeg",0)
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

print("[Info] Training size :{}\nTraining label size {}\n  ".format(X_train.shape,len(y_train)))

if args["load_model"] == 0:
	model = MovingBlurCnn.build(X_train.shape[1], X_train.shape[2], X_train.shape[3])
	model.compile(loss = 'mean_absolute_error',optimizer = 'rmsprop')
	model.fit(X_train,y_train,nb_epoch = nb_epoch,batch_size = 5)
	if args["save_model"] == 1:
		model_json = model.to_json()
		with open("models/" + args["path"]+".json","w") as json_file:
			json_file.write(model_json)
		model.save_weights("models/" + args["path"]+".h5", overwrite=True)
 
else:
	json_file = open("models/" + args["path"]+".json",'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights("models/" + args["path"]+".h5")

print(model.summary())


##################################################################################################################################################################
# Show results
prediction = model.predict(X_test) 
print(np.c_[prediction,y_test])
print(np.sqrt(mse(prediction,y_test)))

k = sorted(range(len(prediction)), key=lambda k: prediction[k])
print(k)

fig = plt.figure(figsize=(10,6.5))
ax_list = fig.axes
print(ax_list)
plt.scatter(np.arange(0,len(k)),y_test[k],c ='b')
plt.scatter(np.arange(0,len(k)),prediction[k],c ='r')
plt.legend(("Ground truth","Predictions"))
plt.ylabel("Degrees")

#cv2.putText(fig, "test", (5, 20),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


plt.show()





