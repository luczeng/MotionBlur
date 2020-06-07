import torch.nn as nn
import torch.nn.functional as F
import copy


class MotionNet(nn.Module):
    def __init__(self, n_layers: int, n_sublayers: int, n_features: int, img_shape: list):
        """
            Network layer definition

            :param n_layers                 number of layers
            :param n_sublayers              number of sublayers (within each layers)
            :param n_features               number of features after the first layer
            :param img_shape                [well you should be sleeping now thenny, nx]

            TODO: evaluate which init is best
        """

        super(MotionNet, self).__init__()

        # Store args
        self.n_layers = n_layers
        self.n_sublayers = n_sublayers
        self.n_features = n_features
        self.img_shape = img_shape
        self.output_img_shape = self._compute_conv_size()
        self.dense_layer_size = self.output_img_shape[0] * self.output_img_shape[1] * n_features * n_layers

        # Activation and pool
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU()

        # Convolutional
        self.convolutional = nn.ModuleList()

        # First layer
        layer = nn.ModuleList()
        layer.append(nn.Conv2d(1, n_features, 3))
        for sublayer in range(1, self.n_sublayers):
            layer.append(nn.Conv2d(n_features, n_features, 3))
        self.convolutional.append(layer)

        # Other convolutional layers
        for k in range(1, n_layers):
            layer = nn.ModuleList()
            layer.append(nn.Conv2d(n_features * k, n_features * (k + 1), 3))
            for sublayer in range(1, self.n_sublayers):
                layer.append(nn.Conv2d(n_features * (k + 1), n_features * (k + 1), 3))
            self.convolutional.append(layer)

        # Classifier
        self.lin1 = nn.Linear(self.dense_layer_size, 256)
        self.lin2 = nn.Linear(256, 2)

    def _one_pass(self, x):
        """
            Pass of the net on one image

            :param x input tensor
        """

        for layer in self.convolutional:
            for sublayer in layer:
                x = F.relu(sublayer(x))
            x = self.pool(x)

        x = x.view(-1, self.dense_layer_size)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

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
