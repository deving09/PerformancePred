import torch
import torch.nn as nn
from torchvision import models
import logging
from itertools import chain

import torch.nn.functional as F

from .basenet import *
from .alexnet import AlexNet
from .vgg import VGG
from .resnet import ResNet
from .densenet import DenseNet

_MODELS = {}

_MODELS.update(alexnet._MODELS)

_MODELS.update(vgg._MODELS)

_MODELS.update(resnet._MODELS)

_MODELS.update(densenet._MODELS)



def _load_base_model(model_type, layer_probe="penultimate", pretrained=True, num_classes=None):
    if model_type in _MODELS:
        net =  _MODELS[model_type](model_type, layer_probe=layer_probe, pretrained=pretrained, num_classes=num_classes)
        return net
    else:
        raise ValueError("Model Type: %s is not specified" % model_type)

