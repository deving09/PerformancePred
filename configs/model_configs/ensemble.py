import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "res-ensemble",
        layer_probe = "penultimate",
        pretrained = True
)

