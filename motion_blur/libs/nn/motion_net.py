import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


class MotionNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_sublayers: int,
        n_features: int,
        img_shape: list,
        as_gray: bool,
        regression: bool = False,
        n_angles: int = None,
        n_lengths: int = None,
    ):
        """
            Network layer definition

            :param n_layers                 number of layers
            :param n_sublayers              number of sublayers (within each layers)
            :param n_features               number of features after the first layer
            :param img_shape                [ny, nx]
            :param as_gray                  True for grayscale images
            :param regression               True for regression network (2 outputs)
            :param n_angles                 number of angles
            :param n_lengths                number of lengths

            TODO: refactor inputs with config file
            TODO: set init
            TODO: remove img shape as input? Or make it optional?
        """

        super(MotionNet, self).__init__()

        # Store args
        self.n_layers = n_layers
        self.n_sublayers = n_sublayers
        self.n_features = n_features
        self.img_shape = img_shape
        self.output_img_shape = self._compute_conv_size()
        self.dense_layer_size = (
            self.output_img_shape[0] * self.output_img_shape[1] * n_features * n_layers
        )  # TODO: remove

        # Activation and pool
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        # Convolutional
        self.convolutional = nn.ModuleList()

        # First layer
        if as_gray:
            self.feature_size_in = 1
        else:
            self.feature_size_in = 3

        layer = nn.ModuleList()
        layer.append(nn.Conv2d(self.feature_size_in, n_features, 3))
        for sublayer in range(1, self.n_sublayers):
            layer.append(nn.Conv2d(n_features, n_features, 3))
        self.convolutional.append(layer)

        # Other convolutional layers
        for k in range(0, n_layers - 1):
            layer = nn.ModuleList()
            layer.append(nn.Conv2d(n_features * 2 ** k, n_features * 2 ** (k + 1), 3))
            for sublayer in range(1, self.n_sublayers):
                layer.append(nn.Conv2d(n_features * 2 ** (k + 1), n_features * 2 ** (k + 1), 3))
            self.convolutional.append(layer)

        # Classifier (2 inputs for regression, n_angles + n_lengths for classification)
        if regression:
            self.lin1 = nn.Linear(n_features * 2 ** (k + 1), 2)
        else:
            if n_angles and n_lengths:
                # self.lin1 = nn.Linear(n_features * 2 ** (k + 1), n_angles + n_lengths)
                self.lin1 = nn.Linear(n_features * 2 ** (k + 1), n_angles)
            else:
                raise ValueError("Speficy number of angles for classification net")

    def _one_pass(self, x):
        """
            Pass of the net on one image

            :param x input tensor
        """

        for layer in self.convolutional:
            for sublayer in layer:
                x = F.relu(sublayer(x))
            x = self.pool(x)

        # Global average pooling
        GlobalAvgPool = nn.AvgPool2d((x.shape[2], x.shape[3]))  # find an alternative to this mb?
        x = GlobalAvgPool(x)
        x = x.view(-1, x.shape[1])

        x = self.lin1(x)

        return x

    def forward(self, input):
        """
            Pass of the net over a batch

            :param input tensor or list of tensors
        """

        if type(input) == list:
            scores = []
            for i, x in enumerate(input):
                score = self._one_pass(x)
                scores.append(score)
        else:
            scores = self._one_pass(input)

        return scores

    def _compute_conv_size(self) -> list:
        """
            Computes the img size of the tensor at the end of the convolutional layers
        """

        conv_shape = copy.deepcopy(self.img_shape)

        for lay in range(self.n_layers):
            for sublay in range(self.n_sublayers):
                conv_shape[0] -= 2
                conv_shape[1] -= 2

            conv_shape[0] = conv_shape[0] // 2
            conv_shape[1] = conv_shape[1] // 2

        return conv_shape
