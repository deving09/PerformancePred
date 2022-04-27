import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision
import numpy as np

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataset import Dataset


import logging

from .data import *


class CIFAR10_C(Dataset):

    _repr_indent = 4

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None):
        self.root = root
        self.name = name
        self.sub_name = sub_name
        self.transform = transform
        self.target_tranform = target_transform


        self.data = np.load(self.root)
        fn_tokens = self.root.split("/")
        fn_tokens[-1] = "labels.npy"
        label_fn = "/".join(fn_tokens)
        self.labels = np.load(label_fn)


        if sub_name:
            sub_split = int(sub_name)
            self.data = self.data[(sub_split - 1) * 50000: sub_split * 50000]
            self.labels = self.labels[(sub_split - 1) * 50000: sub_split * 50000]

        self.len = len(self.labels)


        self.class_samples = {}

        for i, class_idx in enumerate(self.labels):
            if type(class_idx) != int:
                class_idx = class_idx.item()

            if class_idx in self.class_samples:
                self.class_samples[class_idx].append(i)
            else:
                self.class_samples[class_idx] = [i]

        self.classes = list(self.class_samples.keys()) 

    
    def __getitem__(self, index):
        x = self.transform(self.data[index])
        y = np.long(self.labels[index])
        return x, y

    def __len__(self):
        return self.len
    
    
    def get_base_classes_ids(self):
        return set(range(len(self.class_samples.keys())))


    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.root is not None:
            body.append(f"Root location: {self.root}")

        if self.name is not None:
            body.append(f"Dataset Name: {self.name}")

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)





class CIFAR10_1(Dataset):

    _repr_indent = 4

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None):

        self.transform = transform
        
        self.root = root
        data_fn = root + "_data.npy"
        labels_fn = root + "_labels.npy"

        self.data =  np.load(data_fn)
        self.labels = np.load(labels_fn)

        logging.info(self.data.shape)
        logging.info("CIFAR 10.1")

        self.name = name
        self.sub_name = sub_name

        self.len = len(self.labels)


        self.class_samples = {}

        for i, class_idx in enumerate(self.labels):
            if type(class_idx) != int:
                class_idx = class_idx.item()

            if class_idx in self.class_samples:
                self.class_samples[class_idx].append(i)
            else:
                self.class_samples[class_idx] = [i]

        self.classes = list(self.class_samples.keys()) 
        logging.info(self.classes)


    def __getitem__(self, index):
        x = self.transform(self.data[index]) 
        y = np.long(self.labels[index])
        return x, y

    def __len__(self):
        return self.len
    
    
    def get_base_classes_ids(self):
        return set(range(len(self.class_samples.keys())))

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.root is not None:
            body.append(f"Root location: {self.root}")

        if self.name is not None:
            body.append(f"Dataset Name: {self.name}")

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class CIFAR10(tvdatasets.CIFAR10):


    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None, train=True):
        super(CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform, 
                                        train=train)

        
        self.name = name
        self.sub_name = sub_name

        self.samples = self.data
        
        self.class_samples = {}
        for i, class_idx in enumerate(self.targets):
            if type(class_idx) != int: 
                class_idx = class_idx.item()
            if class_idx in self.class_samples:
                self.class_samples[class_idx].append(i)
            else:
                self.class_samples[class_idx] = [i]
        
        logging.info(self.classes)
        logging.info(self.class_to_idx)
        logging.info(list(self.class_samples.keys()))
        self.loader = tvdatasets.folder.default_loader

    def __len__(self):
        return len(self.samples)
   
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target 
    
    def get_base_classes_ids(self):
        return set(range(len(self.classes)))



class CIFAR100_C(Dataset):
    _repr_indent = 4

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None):
        self.root = root
        self.name = name
        self.sub_name = sub_name
        self.transform = transform
        self.target_tranform = target_transform


        self.data = np.load(self.root)

        fn_tokens = self.root.split("/")
        fn_tokens[-1] = "labels.npy"
        label_fn = "/".join(fn_tokens)
        self.labels = np.load(label_fn)

        self.len = len(self.labels)


        self.class_samples = {}

        for i, class_idx in enumerate(self.labels):
            if type(class_idx) != int:
                class_idx = class_idx.item()

            if class_idx in self.class_samples:
                self.class_samples[class_idx].append(i)
            else:
                self.class_samples[class_idx] = [i]

        self.classes = list(self.class_samples.keys()) 

    
    def __getitem__(self, index):
        x = self.transform(self.data[index]) 
        y = np.long(self.labels[index])
        return x, y

    def __len__(self):
        return self.len
    
    
    def get_base_classes_ids(self):
        return set(range(len(self.class_samples.keys())))


    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.root is not None:
            body.append(f"Root location: {self.root}")

        if self.name is not None:
            body.append(f"Dataset Name: {self.name}")

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)




class CIFAR100(tvdatasets.CIFAR100):


    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None, train=True):
        super(CIFAR100, self).__init__(root, transform=transform,
                                        target_transform=target_transform, 
                                        train=train)
        
        self.name = name
        self.sub_name = sub_name

        self.samples = self.data
        self.class_samples = {}
        for i, class_idx in enumerate(self.targets):
            if type(class_idx) != int: 
                class_idx = class_idx.item()
            if class_idx in self.class_samples:
                self.class_samples[class_idx].append(i)
            else:
                self.class_samples[class_idx] = [i]

        
        self.loader = tvdatasets.folder.default_loader

        logging.info("len og 100: %d" %self.__len__())

    def __len__(self):
        return len(self.samples)
   
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target 
    
    def get_base_classes_ids(self):
        return set(range(len(self.classes)))
    
