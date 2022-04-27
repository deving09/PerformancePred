import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "resnet34",
        layer_probe = "penultimate",
        pretrained = True
)

