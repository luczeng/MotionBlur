from skimage import io
from motion_blur.libs.forward_models.kernels.motion import motion_kernel
from motion_blur.libs.forward_models.linops.convolution import Convolution
import torch


def evaluate_one_image_regression(net, img_path, net_type, n_angles=60, L_min=0, L_max=10, as_gray=True):
    """
        Evaluate the linear model on one image using the distance to the true value
        Returns the average angle error

        TODO: this is not modular enough
    """

    img = io.imread(img_path, as_gray=as_gray)

    # Prepare angle and length lists
    angles = torch.linspace(0, 180, n_angles)
    if L_min != L_max:
        lengths = torch.arange(L_min, L_max, 2)  # odd values
    else:
        lengths = [torch.tensor([L_min]).float()]

    angle_loss, length_loss = 0, 0
    for angle in angles:
        for L in lengths:
            # Blur image
            kernel = motion_kernel(angle, int(L))
            H = Convolution(kernel)
            blurred_img = torch.tensor(H * img)

            # Cast into net format
            blurred_img = blurred_img.type(net_type)  # to have an image between 0 - 255
            blurred_img = blurred_img.permute(2, 0, 1)
            blurred_img = blurred_img[None, :, :, :]

            # Infer
            x = net.forward(blurred_img)

            # Calculate error
            angle_loss += torch.abs(x[0, 0] - angle).detach()
            length_loss += torch.abs(x[0, 1] - L).detach()

    angle_loss /= n_angles * len(lengths)
    length_loss /= n_angles * len(lengths)

    return angle_loss, length_loss


def evaluate_one_image_classification(net, img_path, net_type, n_angles=60, L_min=0, L_max=10, as_gray=True):
    """
        Evaluate the linear model on one image using the distance to the true value
        Returns the average angle error

        TODO: this is not modular enough
    """

    img = io.imread(img_path, as_gray=as_gray)

    angle_list = torch.linspace(0, 180, n_angles).float()  # odd values
    L = L_min

    angle_loss = 0
    for angle in angle_list:

        # Blur image
        kernel = motion_kernel(angle, int(L))
        H = Convolution(kernel)
        blurred_img = torch.tensor(H * img)

        # Cast into net format
        blurred_img = blurred_img.type(net_type)
        if as_gray:
            blurred_img = blurred_img[None, None, :, :]
        else:
            blurred_img = blurred_img.permute(2, 0, 1)
            blurred_img = blurred_img[None, :, :, :]

        # Infer
        x = net.forward(blurred_img)

        # Calculate erro
        angle_loss += torch.abs(angle_list[torch.argmax(x)] - angle).detach()

    angle_loss /= n_angles

    return angle_loss
