import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "resnet101",
        layer_probe = "penultimate",
        pretrained = True
)

