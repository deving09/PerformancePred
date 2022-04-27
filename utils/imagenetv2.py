import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *




class ImageNetV2Dataset(tvdatasets.vision.VisionDataset):


    def __init__(self, root, transform=None, target_transform=None, is_valid_file=None,
            name=None, sub_name=None):
        super(ImageNetV2Dataset, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        classes, class_to_idx = self._find_classes(self.root)
        
        self.name = name
        self.sub_name = sub_name

        extensions = tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None

        samples = tvdatasets.folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
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
    
    def get_base_classes_ids(self):
        return set(range(len(self.classes)))
    
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]

        classes = [ int(c) if c.isdigit() else c for c in classes]
        classes.sort()
        classes = [str(c) for c in classes]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx 
