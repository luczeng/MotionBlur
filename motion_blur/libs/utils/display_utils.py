import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import numpy as np
from motion_blur.libs.forward_models.kernels.motion import MotionKernel
from motion_blur.libs.forward_models.linops.convolution import Convolution

class Formatter(object):
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return "x={:.01f}, y={:.01f}, z={:.01f}".format(int(x), int(y), z)


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


class Rotations:
    """ Helper class to generate rotation kernel at several angles and length"""
    def __init__(self, image, L, NAngles):
        self.image = image
        self.NAngles = NAngles
        self.L = L
        self.Angles = sorted(np.random.uniform(0, 180, NAngles))

    def Apply(self):
        self.Out = np.zeros((self.image.shape[0], self.image.shape[1], self.NAngles))
        self.Kernels = [None]*self.NAngles

        for i in range(self.NAngles):
            self.Kernels[i] = Convolution(MotionKernel(self.Angles[i], self.L))
            self.Out[:, :, i] = self.Kernels[i]*self.image
