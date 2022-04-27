import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "densenet161",
        layer_probe = "penultimate",
        pretrained = True
)

