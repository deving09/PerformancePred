import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision
import logging


from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *

from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataset import Dataset

import numpy as np
import pandas as pd

class BootstrapDataset(TensorDataset):

    def __init__(self, *tensors, name=None, class_sublist=None, 
            sub_name=None, root=None, dataset=None):

        self.name = name
        self.class_sublist = class_sublist
        self.sub_name = sub_name
        self.root = root
        self.dataset=dataset

        super(BootstrapDataset, self).__init__(*tensors)

        self.class_samples = build_class_samples_idx(zip(self.tensors[0], self.tensors[1]))


    def get_base_classes_ids(self):
        return self.dataset.get_base_classes_ids()

    def base2instance(self):
        return self.dataset.base2instance()

    def instance2base(self):
        return self.dataset.instance2base()

    def base_filter(self):

        if hasattr(self.dataset, "base_filter"):
            return self.dataset.base_filter()
        else:
            return None

    def base_conversion(self):
        if hasattr(self.dataset, "base_conversion"):
            return self.dataset.base_conversion()
        else:
            return None

    def target_conversion(self):
        if hasattr(self.dataset, "target_conversion"):
            return self.dataset.target_conversion()
        else:
            None
    
    def get_base_classes_names(self):
        return self.dataset.get_base_classes_names()




class BootstrapStoreDataset(Dataset):

    def __init__(self, *tensors, name=None, class_sublist=None, 
            sub_name=None, root=None, dataset=None,
            filename=None, save=False):

        self.name = name
        self.class_sublist = class_sublist
        self.sub_name = sub_name
        self.root = root
        self.dataset=dataset
        self.filename = filename
        self.save = save

        x = tensors[0].numpy()
        y = tensors[1]
        
        if self.save:
            y_numpy = y.numpy()

            np.save(self.filename + "_y.npy", y_numpy)
            np.save(self.filename + ".npy", x)

        self.y = y

        self.fp = np.memmap(self.filename, dtype='float32', mode='w+',
                shape=x.shape)

        self.fp[:] = x[:]
        self.fp.flush()

        self.fp = np.memmap(self.filename, dtype='float32', mode='r', 
                shape=x.shape)

        self.len = len(y)


        if filename is None:
            raise ValueError("Need Temporary Filename")

        self.class_samples = build_class_samples_idx(zip(tensors[0], tensors[1]))
    
    def __len__(self):
        return self.len


    def __getitem__(self, index):
        x = torch.from_numpy(self.fp[index])
        y = self.y[index]

        return x, y

    def __del__(self):
        del self.fp

        if  isinstance(self.filename, str):
            if os.path.exists(self.filename):
                os.remove(self.filename)
        else:
            self.filename.close()


    def get_base_classes_ids(self):
        return self.dataset.get_base_classes_ids()

    def base2instance(self):
        return self.dataset.base2instance()

    def instance2base(self):
        return self.dataset.instance2base()

    def base_filter(self):

        if hasattr(self.dataset, "base_filter"):
            return self.dataset.base_filter()
        else:
            return None

    def base_conversion(self):
        if hasattr(self.dataset, "base_conversion"):
            return self.dataset.base_conversion()
        else:
            return None

    def target_conversion(self):
        if hasattr(self.dataset, "target_conversion"):
            return self.dataset.target_conversion()
        else:
            None
    
    def get_base_classes_names(self):
        return self.dataset.get_base_classes_names()



class StandardDataset(tvdatasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
            loader=tvdatasets.folder.default_loader, is_valid_file=None,
            name=None, sub_name=None):

        self.name = name
        self.sub_name = sub_name

        super(StandardDataset, self).__init__(root, loader, 
                tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None)

        self.class_samples = build_class_samples_idx(self.samples)


    def base2instance(self):
        return {value:value for key, value in self.class_to_idx.items()}

    def instance2base(self):
        return {value:value for key, value in self.class_to_idx.items()}

    def get_base_classes_ids(self):
        return {value for key, value in self.class_to_idx.items()}

    def get_base_classes_names(self):
        return set(self.classes)


