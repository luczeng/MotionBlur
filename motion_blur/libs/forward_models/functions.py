from numpy.fft import fft2, ifft2
import numpy as np
import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import math

##################################################################################################################################
##################################################################################################################################


class Image:
    def __init__(self, image):
        self.image = image
        self.shape = image.shape

    def LinearBlur(self, theta, L, h):
        # Theta : angle with respects to the vertical axis
        # L 	   : Number of pixels of the blur (odd)

        kernel = np.zeros([L, L])
        pos = int((L - 1) / 2)
        kernel[pos, :] = 1

        out = Image(np.real(ifft2(fft2(self.image) * fft2(h, self.shape))))

        return out


##################################################################################################################################
##################################################################################################################################


def Convolution(In, Kernel):
    # Put checks
    return np.real(ifft2(fft2(In) * fft2(Kernel, In.shape)))


##################################################################################################################################
##################################################################################################################################
def Wiener(In, Kernel, Lambda):
    w = np.conj(fft2(Kernel, In.shape)) / (np.conj(fft2(Kernel, In.shape)) * fft2(Kernel, In.shape) + Lambda)
    out = ifft2(w * fft2(In))
    return np.real(out)


##################################################################################################################################
##################################################################################################################################
def MotionKernel(theta, L):
    kernel = np.zeros([L, L])
    x = np.arange(0, L, 1) - int(L / 2)
    X, Y = np.meshgrid(x, x)

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if np.sqrt(X[i, j] ** 2 + Y[i, j] ** 2) < L / 2:
                kernel[i, j] = LineIntegral(theta, X[i, j] - 0.5, X[i, j] + 0.5, -Y[i, j] - 0.5, -Y[i, j] + 0.5)

    return kernel


##################################################################################################################################
##################################################################################################################################
def LineIntegral(theta, a, b, alpha, beta):
    # Theta : between 0 and 360
    TanTheta = np.tan(np.deg2rad(theta))
    L = 0
    # Checks
    if a > b:
        x = a
        a = b
        b = x
    if alpha > beta:
        x = alpha
        alpha = beta
        beta = x

    if theta != 90 and theta != 270 and theta != 0 and theta != 180:  # non vertical case
        if b >= 0 and beta >= 0 and TanTheta > 0:
            if alpha <= TanTheta * a <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - a) ** 2 + (beta - TanTheta * a) ** 2)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - alpha / TanTheta) ** 2 + (TanTheta * b - alpha) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (beta - alpha) ** 2)

        elif b >= 0 and alpha <= 0 and TanTheta < 0:
            if alpha <= TanTheta * a <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - a) ** 2 + (alpha - TanTheta * a) ** 2)
            elif a <= beta / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - beta / TanTheta) ** 2 + (TanTheta * b - beta) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - beta / TanTheta) ** 2 + (alpha - beta) ** 2)

        elif a <= 0 and beta >= 0 and TanTheta < 0:
            if alpha <= TanTheta * b <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - b) ** 2 + (beta - TanTheta * b) ** 2)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((a - alpha / TanTheta) ** 2 + (TanTheta * a - alpha) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (beta - alpha) ** 2)

        elif a <= 0 and alpha <= 0 and TanTheta > 0:
            if alpha <= TanTheta * b <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - b) ** 2 + (alpha - TanTheta * b) ** 2)
            elif a <= beta / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((a - beta / TanTheta) ** 2 + (TanTheta * a - beta) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - beta / TanTheta) ** 2 + (alpha - beta) ** 2)

        elif a < 0 and b > 0 and alpha < 0 and beta > 0:
            if alpha <= TanTheta * a <= beta:
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (a * TanTheta - b * TanTheta) ** 2)
                else:
                    if TanTheta * a < TanTheta * b:
                        L = np.sqrt((beta / TanTheta - a) ** 2 + (a * TanTheta - beta) ** 2)
                    else:
                        L = np.sqrt((alpha / TanTheta - a) ** 2 + (a * TanTheta - alpha) ** 2)
            else:
                if a <= alpha / TanTheta <= b:
                    if alpha <= TanTheta * b <= beta:
                        L = np.sqrt((b - alpha / TanTheta) ** 2 + (alpha - b * TanTheta) ** 2)
                    else:
                        L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (alpha - beta) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - b) ** 2 + (TanTheta * b - beta) ** 2)

    else:
        if theta == 90 or theta == 270:
            if a < 0 and b > 0:
                L = (beta - alpha) * (b - a)
        else:
            if alpha < 0 and beta > 0:
                L = (beta - alpha) * (b - a)

    return L


##################################################################################################################################
##################################################################################################################################


class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return "x={:.01f}, y={:.01f}, z={:.01f}".format(int(x), int(y), z)


##################################################################################################################################
##################################################################################################################################


class Rotations:
    def __init__(self, image, L, NAngles):
        self.image = image
        self.NAngles = NAngles
        self.L = L
        self.Angles = sorted(np.random.uniform(0, 180, NAngles))

    def Apply(self):
        """
		self.Out = [None]*self.NAngles
		self.Kernels = [None]*self.NAngles
		for i in range(self.NAngles):
			self.Kernels[i] = MotionKernel(self.Angles[i] , self.L)
			self.Out[i] = Convolution(self.image, self.Kernels[i])
		"""
        self.Out = np.zeros((self.image.shape[0], self.image.shape[1], self.NAngles))
        self.Kernels = np.zeros((self.L, self.L, self.NAngles))
        print(self.Kernels.shape)
        for i in range(self.NAngles):
            self.Kernels[:, :, i] = MotionKernel(self.Angles[i], self.L)
            self.Out[:, :, i] = Convolution(self.image, self.Kernels[:, :, i])


##################################################################################################################################
##################################################################################################################################


import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1


class PageSlider(matplotlib.widgets.Slider):
    def __init__(
        self, ax, label, numpages=10, valinit=0, valfmt="%1d", closedmin=True, closedmax=True, dragging=True, **kwargs
    ):

        self.facecolor = kwargs.get("facecolor", "w")
        self.activecolor = kwargs.pop("activecolor", "b")
        self.fontsize = kwargs.pop("fontsize", 10)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages, valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i == valinit else self.facecolor
            r = matplotlib.patches.Rectangle(
                (float(i) / numpages, 0), 1.0 / numpages, 1, transform=ax.transAxes, facecolor=facecolor
            )
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(
                float(i) / numpages + 0.5 / numpages,
                0.5,
                str(i + 1),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.fontsize,
            )
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(
            bax, label="$\u25C0$", color=self.facecolor, hovercolor=self.activecolor
        )
        self.button_forward = matplotlib.widgets.Button(
            fax, label="$\u25B6$", color=self.facecolor, hovercolor=self.activecolor
        )
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i + 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i - 1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)


##################################################################################################################################
##################################################################################################################################
def vector_coord(angle, length):
    cartesianAngleRadians = (450 - angle) * math.pi / 180.0
    x = length * math.cos(cartesianAngleRadians)
    y = length * math.sin(cartesianAngleRadians)
    return x, y
