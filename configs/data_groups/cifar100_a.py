import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
    ])

valid_transform = transforms.Compose([
    transforms.ToTensor()
    ])


config = dict(
        base = [("cifar100", transform, valid_transform, None)],
        cal  = [
         ("cifar100c-jpeg-compression", transform, valid_transform, None),
         ("cifar100c-frost", transform, valid_transform, None),
         ("cifar100c-elastic-transform", transform, valid_transform, None),
         ("cifar100c-contrast", transform, valid_transform, None),
         ("cifar100c-pixelate", transform, valid_transform, None),
         ("cifar100c-gaussian-noise", transform, valid_transform, None),
         ("cifar100c-impulse-noise", transform, valid_transform, None),
         ("cifar100c-shot-noise", transform, valid_transform, None),
         ("cifar100c-fog", transform, valid_transform, None),
         ("cifar100c-snow", transform, valid_transform, None),
         ("cifar100c-brightness", transform, valid_transform, None)],
        test = [ 
         ("cifar100c-motion-blur", transform, valid_transform, None),
         ("cifar100c-defocus-blur", transform, valid_transform, None),
         ("cifar100c-glass-blur", transform, valid_transform, None),
         ("cifar100c-zoom-blur", transform, valid_transform, None),
         ("cifar100c-spatter", transform, valid_transform, None),
         ("cifar100c-speckle-noise", transform, valid_transform, None),
         ("cifar100c-gaussian-blur", transform, valid_transform, None)
         #("cifar100.1-v6", transform, valid_transform, None),
         #("cifar100.1-v4", transform, valid_transform, None)
         ]
 )

