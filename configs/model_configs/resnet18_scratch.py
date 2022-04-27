import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "resnet18",
        layer_probe = "penultimate",
        pretrained = False
)

