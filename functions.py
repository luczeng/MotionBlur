from numpy.fft import fft2,ifft2
import numpy as np

##################################################################################################################################
##################################################################################################################################

class Image:
	def __init__(self,image):
		self.image = image
		self.shape = image.shape


	def LinearBlur(self,theta,L,h):
	#Theta : angle with respects to the vertical axis
	#L 	   : Number of pixels of the blur (odd)

		kernel = np.zeros([L,L])
		pos = int((L-1)/2)
		kernel[pos,:] = 1
		
		out = Image(np.real(ifft2(fft2(self.image)*fft2(h,self.shape))))

		return out
##################################################################################################################################
##################################################################################################################################

def Wiener(In,Kernel,Lambda):
	w = np.conj(fft2(Kernel,In.shape)) / ( np.conj(fft2(Kernel,In.shape))* fft2(Kernel,In.shape) + Lambda)
	out = ifft2(  w*fft2(In) )
	return np.real(out)

##################################################################################################################################
##################################################################################################################################
def MotionBlur(theta,L):
	kernel = np.zeros([L,L])
	x = np.arange(0,L,1) - int(L/2)
	X,Y = np.meshgrid(x,x)

	for i in range(x.shape[0]):
		for j in range(x.shape[0]):
			if np.sqrt(X[i,j]**2 + Y[i,j]**2)<L/2:
				kernel[i,j] = LineIntegral(theta,X[i,j]-0.5 ,X[i,j]+0.5 ,-Y[i,j]-0.5 ,-Y[i,j]+0.5 )

	return kernel

##################################################################################################################################
##################################################################################################################################
def LineIntegral(theta,a,b,alpha,beta):
	#Theta : between 0 and 360
	TanTheta = np.tan(np.deg2rad(theta))
	L =0
	#Checks
	if a>b:
		x=a
		a=b
		b=x
	if alpha>beta:
		x = alpha
		alpha =beta
		beta = x

	if (theta != 90 and theta != 270 and theta!= 0 and theta != 180): 		#non vertical case
		if (a>=0 and alpha>=0):
			if alpha <= TanTheta*a <= beta: 	#pointing upward, case 1
				if alpha<=TanTheta*b<=beta:
					L = np.sqrt( (b-a)**2 + (TanTheta*b - TanTheta*a)**2)
				else:
					L = np.sqrt( (beta/TanTheta-a)**2 + (beta - TanTheta*a)**2)
			elif a<=alpha/TanTheta<=b:			#pointing upward, case 2
				if alpha<=TanTheta*b<=beta:
					L = np.sqrt( (b-alpha/TanTheta)**2 + (TanTheta*b - alpha)**2)
				else:
					L = np.sqrt( (beta/TanTheta-alpha/TanTheta)**2 + (beta - alpha)**2)

		elif (a>=0 and beta<=0):
			if alpha <= TanTheta*a <= beta: 	#pointing downward, case 1
				if alpha<=TanTheta*b<=beta:
					L = np.sqrt( (b-a)**2 + (TanTheta*b - TanTheta*a)**2)
				else:
					L = np.sqrt( (alpha/TanTheta-a)**2 + (alpha - TanTheta*a)**2)
			elif a<=alpha/TanTheta<=b:			#pointing downward, case 2
				if alpha<=TanTheta*b<=beta:
					L = np.sqrt( (b-beta/TanTheta)**2 + (TanTheta*b - beta)**2)
				else:
					L = np.sqrt( (alpha/TanTheta-beta/TanTheta)**2 + (alpha - beta)**2)

		elif (a<=0 and alpha>=0):
			if alpha <= TanTheta*b <= beta: 	#pointing upward, case 1
				if alpha<=TanTheta*a<=beta:
					L = np.sqrt( (b-a)**2 + (TanTheta*b - TanTheta*a)**2)
				else:
					L = np.sqrt( (beta/TanTheta-b)**2 + (beta - TanTheta*b)**2)
			elif a<=alpha/TanTheta<=b:			#pointing upward, case 2
				if alpha<=TanTheta*a<=beta:
					L = np.sqrt( (a-alpha/TanTheta)**2 + (TanTheta*a - alpha)**2)
				else:
					L = np.sqrt( (beta/TanTheta-alpha/TanTheta)**2 + (beta - alpha)**2)

		elif (b<=0 and beta<=0):
			if alpha <= TanTheta*b <= beta: 	#pointing downward, case 1
				if alpha<=TanTheta*a<=beta:
					L = np.sqrt( (b-a)**2 + (TanTheta*b - TanTheta*a)**2)
				else:
					L = np.sqrt( (alpha/TanTheta-b)**2 + (alpha - TanTheta*b)**2)
			elif a<=beta/TanTheta<=b:			#pointing downward, case 2
				if alpha<=TanTheta*a<=beta:
					L = np.sqrt( (a-beta/TanTheta)**2 + (TanTheta*a - beta)**2)
				else:
					L = np.sqrt( (alpha/TanTheta-beta/TanTheta)**2 + (alpha - beta)**2)
		
		elif (a<0 and b>0 and alpha<0 and beta>0):
			if alpha<=TanTheta*a<=beta:
				if alpha<=TanTheta*b<=beta:
					L = np.sqrt((b-a)**2+(a*TanTheta - b*TanTheta)**2)
				else:
					if (TanTheta*a<TanTheta*b):
						L = np.sqrt((beta/TanTheta-a)**2+(a*TanTheta - beta)**2)
					else:
						L = np.sqrt((alpha/TanTheta-a)**2+(a*TanTheta - alpha)**2)
			else:
				if a<=alpha/TanTheta<=b:
					if alpha<=TanTheta*b<=beta:
						L = np.sqrt((b-alpha/TanTheta)**2+(alpha - b*TanTheta)**2)
					else:
						L = np.sqrt((beta/TanTheta-alpha/TanTheta)**2+(alpha - beta)**2)
				else:
						L = np.sqrt((beta/TanTheta-b)**2+(TanTheta*b - beta)**2)
			
	else:
		if (theta == 90 or theta == 270):
			if (a<0 and b>0):
				L = (beta-alpha)*(b-a)
		else:
			if (alpha<0 and beta>0):
				L = (beta-alpha)*(b-a)

	return L

##################################################################################################################################
##################################################################################################################################

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(int(x), int(y), z)

