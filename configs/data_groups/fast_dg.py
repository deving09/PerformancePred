import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

config = dict(
        base = [("imagenet", transform, valid_transform, None)],
        cal  = [("imagenetc-digital-jpeg", transform, valid_transform, None),
         ("imagenet-r", transform, valid_transform, None),
         ("imagenet-a", transform, valid_transform, None)],
        test = [ #("imagenetv2-top", transform, None),
         ("imagenetv2-top-fixed", transform, valid_transform, None),
         ("imagenet-vid", transform, valid_transform, None),
         ("imagenetc-extra-saturate", transform, valid_transform, None)],
 )

