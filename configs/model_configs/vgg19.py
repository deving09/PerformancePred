import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "vgg19",
        layer_probe = "penultimate",
        pretrained = True
)

