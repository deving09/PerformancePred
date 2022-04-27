import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
    ])

valid_transform = transforms.Compose([
    transforms.ToTensor()
    ])

config = dict(
        base = [("cifar10c-shot-noise", transform, valid_transform, None)],
        cal  = [
         ("cifar10", transform, valid_transform, None),
         ("cifar10c-shot-noise", transform, valid_transform, None),
         ("cifar10c-brightness", transform, valid_transform, None),
         ("cifar10c-defocus-blur", transform, valid_transform, None),
         ("cifar10.1-v6", transform, valid_transform, None),
         ("cifar10.1-v4", transform, valid_transform, None)],
        test = [ #("imagenetv2-top", transform, None),
         ("cifar10", transform, valid_transform, None),
         ("cifar10.1-v6", transform, valid_transform, None),
         ("cifar10.1-v4", transform, valid_transform, None)],
 )

