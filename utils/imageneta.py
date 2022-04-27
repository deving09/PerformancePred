import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *


class ImageNetADataset(tvdatasets.DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
            loader=tvdatasets.folder.default_loader, is_valid_file=None,
            name=None, sub_name=None):
        super(ImageNetADataset, self).__init__(root, loader=loader,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file,
                extensions= tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None)

        
        self.name = name
        self.sub_name = sub_name
        self.class_samples = build_class_samples_idx(self.samples)

        self.class_sublist = [
                6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 
                113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 
                310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 
                402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 
                514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 
                640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 
                779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 
                888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 
                986, 987, 988]

        self.class_sublist_tensor = torch.tensor(self.class_sublist)
    
    def base2instance(self):
        return {val: i for i, val in enumerate(self.class_sublist)}

    def instance2base(self):
        return {i:val for i, val in enumerate(self.class_sublist)}

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



    def get_base_classes_ids(self):
        return set(self.class_sublist)

    def get_base_classes_names(self):
        return set(self.classes)

