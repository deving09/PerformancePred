import os
import sys

import json
import torch
import torch.utils.data as data_utils
import torchvision

from torchvision.datasets import vision
import torchvision.datasets as tvdatasets

from .data import *

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import pathlib
import logging

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions) 

class ImageNetVidRobustDataset(tvdatasets.DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None,
            loader=tvdatasets.folder.default_loader, is_valid_file=None,
            name=None, sub_name=None):
        super(ImageNetVidRobustDataset, self).__init__(root, loader=loader,
                transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file,
                extensions= tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None)

        extensions= tvdatasets.folder.IMG_EXTENSIONS if is_valid_file is None else None

        self.name = name
        self.sub_name = sub_name
        
        with open(os.path.join(self.root, "../misc/wnid_map.json")) as f:
            self.cn_to_vid_cn = json.loads(f.read())
        with open(os.path.join(self.root, "../misc/rev_wnid_map.json")) as f:
            self.vid_cn_to_cn = json.loads(f.read())
        with open(os.path.join(self.root, "../misc/imagenet_vid_class_index.json")) as f:
            self.vid_idx_to_vid_cn = json.loads(f.read())
        with open(os.path.join(self.root, "../misc/imagenet_class_index.json")) as f:
            self.idx_to_cn = json.loads(f.read())
        with open(os.path.join(self.root, "../metadata/big_labels.json")) as f:
            self.labels_paths = json.loads(f.read())


        self.cn_to_idx = {v[0]: k for k, v in self.idx_to_cn.items()}

        self.vid_cn_to_vid_idx = {v[0]: k for k, v in self.vid_idx_to_vid_cn.items()}

        self.idx_to_vid_idx = {int(self.cn_to_idx[k]): int(self.vid_cn_to_vid_idx[v]) for k, v in self.cn_to_vid_cn.items()}

        folders, folders_to_idx = self._find_folders(self.root)

        self.folders = folders
        self.folders_to_idx = folders_to_idx

        self.class_to_idx = {self.vid_idx_to_vid_cn[k][0]: k for k in sorted(list(self.vid_idx_to_vid_cn.keys())) }

        self.classes = list(self.class_to_idx.keys())
        self.num_classes = len(self.classes)

        self.vid_idx_to_idx_list = {}
        for k, v in self.idx_to_vid_idx.items():
            if v in self.vid_idx_to_idx_list:
                self.vid_idx_to_idx_list[v].append(k)
            else:
                self.vid_idx_to_idx_list[v] = [k]

        self.vid_idx_to_idx_list = {k: torch.tensor(v) for k, v in self.vid_idx_to_idx_list.items()}

        samples = self._make_dataset(self.root, self.folders_to_idx,
                self.labels_paths,
                extensions, is_valid_file)

        self.samples = samples

        self.targets = [s[1] for s in samples]
        self.class_samples = build_class_samples_idx(self.samples)

        self.class_sublist = list(self.idx_to_vid_idx.keys())

        self.class_sublist_tensor = torch.tensor(self.class_sublist)
    

    def _find_folders(self, dr):
        classes = [d.name for d in os.scandir(dr) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, 
            directory,
            class_to_idx,
            path_to_label,
            extensions=None,
            is_valid_file=None):

        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None

        if both_none or both_something:
            raise ValueError("Either extensions or is_valid_file must be set")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir  = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):
                        key_fn = os.path.join("val", target_class, fname)
                        actual_class_idx = path_to_label[key_fn]
                        item = path, actual_class_idx
                        instances.append(item)

        return instances

    def __len__(self):
        return len(self.samples)

    def base2instance(self): 
        return self.idx_to_vid_idx

    def instance2base(self):
        raise ValueError("Not implemented")
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
            new_outputs = torch.zeros((outputs.shape[0], self.num_classes))
            for cls_idx, c_list in self.vid_idx_to_idx_list.items():
                new_outputs[:,cls_idx], _ = torch.max(torch.index_select(outputs, 1, c_list), 1)

            targets = torch.tensor([self.idx_to_vid_idx[t.item()] for t in targets])

            return new_outputs, targets
        
        return conv

    def target_conversion(self):

        def conv(outputs, targets):
            new_outputs = torch.zeros((outputs.shape[0], self.num_classes))
            for cls_idx, c_list in self.vid_idx_to_idx_list.items():
                new_outputs[:,cls_idx], _ = torch.max(torch.index_select(outputs, 1, c_list), 1)
            
            return new_outputs, targets

        return conv



    def get_base_classes_ids(self):
        return set(self.class_sublist)

    def get_base_classes_names(self):
        return set(self.classes)

