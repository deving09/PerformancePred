import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets


def build_class_samples_idx(samples):
    c_samples = {}

    for i, (path, class_idx) in enumerate(samples):
        if type(class_idx) != int:
            class_idx = class_idx.item()
        if class_idx in c_samples:
            c_samples[class_idx].append(i)
        else:
            c_samples[class_idx] = [i]

    return c_samples

