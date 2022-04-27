import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "vaishaal-fixed-resnet101",
        layer_probe = "penultimate",
        pretrained = True
)

