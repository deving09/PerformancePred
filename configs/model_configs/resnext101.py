import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "resnext101",
        layer_probe = "penultimate",
        pretrained = True
)

