from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initalize_resnet(num_classes: int, feature_extracting: bool = False, use_pretrained: bool = True):
    """
        Initialises a resnet model to be fine tuned or used for feature extraction
        :param num_classes number of classes
        :feature_extracting true if all weights expect classfier are to be frozen
        :use_pretrained use ImageNet pretrained weights
        :param model
    """

    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extracting)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_classes)

    return model_ft
