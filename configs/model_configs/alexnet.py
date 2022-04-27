import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "alexnet",
        layer_probe = "penultimate",
        pretrained = True
)

