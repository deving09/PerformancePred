import torch.nn as nn
from torchvision import transforms

config = dict(
        net =  "vaishaal-ensemble",
        layer_probe = "penultimate",
        pretrained = True
)

