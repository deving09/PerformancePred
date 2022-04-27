import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "resnet18-cifar",
        layer_probe = "penultimate",
        pretrained = True
)

