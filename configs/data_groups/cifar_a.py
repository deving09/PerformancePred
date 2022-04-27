import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
    ])

valid_transform = transforms.Compose([
    transforms.ToTensor()
    ])


config = dict(
        base = [("cifar10", transform, valid_transform, None)],
        cal  = [
         ("cifar10c-jpeg-compression", transform, valid_transform, None),
         ("cifar10c-frost", transform, valid_transform, None),
         ("cifar10c-elastic-transform", transform, valid_transform, None),
         ("cifar10c-contrast", transform, valid_transform, None),
         ("cifar10c-pixelate", transform, valid_transform, None),
         ("cifar10c-gaussian-noise", transform, valid_transform, None),
         ("cifar10c-impulse-noise", transform, valid_transform, None),
         ("cifar10c-shot-noise", transform, valid_transform, None),
         ("cifar10c-fog", transform, valid_transform, None),
         ("cifar10c-snow", transform, valid_transform, None),
         ("cifar10c-brightness", transform, valid_transform, None)],
        test = [ 
         ("cifar10c-motion-blur", transform, valid_transform, None),
         ("cifar10c-defocus-blur", transform, valid_transform, None),
         ("cifar10c-glass-blur", transform, valid_transform, None),
         ("cifar10c-zoom-blur", transform, valid_transform, None),
         ("cifar10c-spatter", transform, valid_transform, None),
         ("cifar10c-speckle-noise", transform, valid_transform, None),
         ("cifar10c-gaussian-blur", transform, valid_transform, None),
         ("cifar10.1-v6", transform, valid_transform, None),
         ("cifar10.1-v4", transform, valid_transform, None)]
 )

