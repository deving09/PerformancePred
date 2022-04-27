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
        cal  = [
         ("imagenetv2-top-fixed", transform, valid_transform, None),
         ("imagenetv2-matched-fixed", transform, valid_transform, None),
         ("imagenetv2-threshold-fixed", transform, valid_transform, None)
         ],
        test = [
         ("imagenetv2-top", transform, valid_transform, None),
         ("imagenetv2-matched", transform, valid_transform, None),
         ("imagenetv2-threshold", transform, valid_transform, None),
         ("imagenet-a", transform, valid_transform, None),
         ("imagenetc-digital-jpeg", transform, valid_transform, None),
         ("imagenetc-weather-frost", transform, valid_transform, None),
         ("imagenetc-digital-elastic", transform, valid_transform, None),
         ("imagenetc-digital-contrast", transform, valid_transform, None),
         ("imagenetc-digital-pixelate", transform, valid_transform, None),
         ("imagenetc-gaussian-noise", transform, valid_transform, None),
         ("imagenetc-impulse-noise", transform, valid_transform, None),
         ("imagenetc-shot-noise", transform,valid_transform,  None),
         ("imagenetc-weather-fog", transform, valid_transform, None),
         ("imagenetc-weather-snow", transform, valid_transform, None),
         ("imagenetc-weather-brightness", transform, valid_transform, None),
         ("imagenetc-motion-blur", transform, valid_transform, None),
         ("imagenetc-defocus-blur", transform, valid_transform, None),
         ("imagenetc-glass-blur", transform, valid_transform, None),
         ("imagenetc-zoom-blur", transform, valid_transform, None),
         ("imagenetc-extra-gaussian", transform, valid_transform, None),
         ("imagenetc-extra-speckle", transform, valid_transform, None),
         ("imagenetc-extra-spatter", transform, valid_transform, None),
         ("imagenetc-extra-saturate", transform, valid_transform, None),
         ("imagenet-r", transform, valid_transform, None),
         ("imagenet-sketch", transform, valid_transform, None),
         ("imagenet-vid", transform, valid_transform, None)
         ],
 )

