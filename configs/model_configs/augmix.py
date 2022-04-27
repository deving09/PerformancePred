import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "augmix",
        layer_probe = "penultimate",
        pretrained = True
)

