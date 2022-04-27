import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *



class ImgCDataset(tvdatasets.vision.VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None):
        super(ImgCDataset, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        classes, class_to_idx = self._find_classes(self.root)

        self.name = name
        self.sub_name = sub_name
        
        extensions = tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None

        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        
        self.class_samples = build_class_samples_idx(self.samples)
        
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
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target 
    
    def _find_classes(self, root):
        """
        Find the class in folders in a dataset
        """

        classes = {root.split("/")[-1]:1 for root, dirnames, fnames in os.walk(root, followlinks=True) if len(dirnames) == 0}
        classes = list(classes.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not  None

        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not  None at the same time")

        if extensions is not None:
            def is_valid_file(x):
                return lambda x: x.lower().endswith(extensions)


        for root, dirnames, fnames in sorted(os.walk(directory, followlinks=True)):
            if len(dirnames) > 0:
                continue

            class_index = class_to_idx[root.split("/")[-1]]
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

        return instances
    
    
    def base2instance(self):
        return {value:value for key, value in self.class_to_idx.items()}

    def instance2base(self):
        return {value:value for key, value in self.class_to_idx.items()}

    def get_base_classes_ids(self):
        return {value for key, value in self.class_to_idx.items()}

    def get_base_classes_names(self):
        return set(self.classes)


