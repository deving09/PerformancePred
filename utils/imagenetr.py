import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *



class ImageNetRDataset(tvdatasets.DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
            loader=tvdatasets.folder.default_loader, is_valid_file=None,
            name=None, sub_name=None):
        super(ImageNetRDataset, self).__init__(root, loader=loader,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file,
                extensions= tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None)

        self.name = name
        self.sub_name = sub_name
        self.class_samples = build_class_samples_idx(self.samples)
        self.class_sublist = [
                1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 
                125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 
                231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 
                296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 
                362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 
                463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 
                637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 
                889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 
                983, 988]
        
        self.class_sublist_tensor = torch.tensor(self.class_sublist)

    def base2instance(self):
        return {val: i for i, val in enumerate(self.class_sublist)}

    def instance2base(self):
        return {i:val for i, val in enumerate(self.class_sublist)}

    def get_base_classes_ids(self):
        return set(self.class_sublist)

    def get_base_classes_names(self):
        return set(self.classes)
    
    def base_filter(self):
        
        def fil(inputs, targets):
            ind_matches = torch.nonzero(torch.tensor([t in self.class_sublist for t in targets])).squeeze()
            inputs = torch.index_select(inputs, 0, ind_matches)
            targets = torch.index_select(targets, 0, ind_matches)
            return inputs, targets
        
        return fil

    def base_conversion(self):

        def conv(outputs, targets):
            outputs = torch.index_select(outputs, 1, self.class_sublist_tensor)
            targets = torch.tensor([self.class_sublist.index(t) for t in targets])
            return outputs, targets
        
        return conv
    
    def target_conversion(self):

        def conv(outputs, targets):
            outputs = torch.index_select(outputs, 1, self.class_sublist_tensor)
            return outputs, targets

        return conv

