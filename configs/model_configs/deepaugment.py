import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "deepaugment",
        layer_probe = "penultimate",
        pretrained = True
)

