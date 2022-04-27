import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import DataLoader, Dataset
import time
from torchvision import transforms

import logging
import os
import json
import collections
import xxhash
#from algorithms.helpers import token_maker

def dedup_datasets(d1, d2, dataset_cls):
    
    # assumes classes
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset1 = dataset_cls(d1, transform=data_transform)
    dataset2 = dataset_cls(d2, transform=data_transform)

    print("Dedup Activated")
    map_1 = {}
    map_2 = {}
    false_1 = {}
    false_2 = {}
    # Slowest possible Code
    for i in range(len(dataset1)):
        x1, y1 = dataset1[i]
        if y1 in map_1:
            map_1[y1].append((x1, i))
        else:
            map_1[y1] = [(x1, i)]

    
    for j in range(len(dataset2)):
        
        if j % 50 == 0:
            print(j)
        x2, y2 = dataset2[j]
        class_matches = map_1[y2]
        for x1, i in class_matches:
            if x2.shape == x1.shape and torch.allclose(x2,x1):
                false_1[i] = 1
                false_2[j] = 1
                break

    idx1 = []
    idx2 = []
    len_d1 = len(dataset1)
    len_d2 = len(dataset2)
    for i in range(max(len_d1, len_d2)):
        if i not in false_1:
            idx1.append(i)

        if i not in false_2:
            idx2.append(i)

    print("Len false 1: %d" %(len(idx1)))
    print("Len false 2: %d" %(len(idx2)))
    return idx1, idx2

def _nested_hash(x):

    if isinstance(x, collections.Hashable):
        
        if callable(x):
            h =  xxhash.xxh64("").hexdigest()
            #logging.info("unhashed X: %s" %(str(x)))
        else:
            h =  xxhash.xxh64(str(x)).hexdigest()
            #logging.info("hashed X: %s\t hash: %s" %(str(x), h))
        return h
    elif isinstance(x, dict):
        l = []
        keys = sorted(list(x.keys()))
        #for key, val in x.items():
        for key in keys:
            val = x[key]
            h = xxhash.xxh64(str(key)).hexdigest()
            #logging.info("hashed X: %s\t hash: %s" %(str(key), h))
            
            l.append(h)
            l.append(_nested_hash(val))
        
        h = xxhash.xxh64(str(tuple(l))).hexdigest()
        return h
    elif isinstance(x, list):
        #logging.info(x)
        #ix = sorted(x)
        l = []
        for v in x:
            l.append(_nested_hash(v))

        l = tuple(l)
        h = xxhash.xxh64(str(l)).hexdigest()
        #logging.info("hashed X: %s\t hash: %s" %(str(l), h))
        return h
    else:
        raise ValueError("Not Supported Hashing: %s" %(str(p)))

    return h


def token_maker(*params):
    base_list = []

    for p in params:
        if isinstance(p, collections.Hashable):
            
            if callable(p):
                #logging.info("Unhashed P: %s" %str(p))
                h =  xxhash.xxh64("").hexdigest()
            else:
                h = xxhash.xxh64(str(p)).hexdigest()
                #logging.info("hashed P: %s\t hash: %s" %(str(p), h))

            base_list.append(h)
        else:
            base_list.append(_nested_hash(p))
    
    #base_list = sorted(base_list)
    h = xxhash.xxh64(str(tuple(base_list))).hexdigest()
    return h

def get_split_filename(names, val_split, test_split, random_seed, method):
    dirname = "data"
    fn = "_".join(names)
    fn += "_%d_%.3f_%.3f_%s.json" %(random_seed, val_split, test_split, method)
    if len(fn) > 255:
        fn = str( xxhash.xxh64(fn).hexdigest()) + ".json"
    fn = os.path.join(dirname, fn)
    return fn


class PseudoSampler(object):

    def __init__(self, dataset, valid_dataset, shuffle=True,
            batch_size=64, random_seed=73, transform=None,
            num_workers=8):

        if isinstance(valid_dataset, list):
            valid_dataset = ConcatDataset(*valid_dataset)
        
        if isinstance(dataset, list):
            dataset = ConcatDataset(*dataset)
        
        self.dataset = dataset
        self.valid_dataset = valid_dataset

        self.train = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=False)


        self.val = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=False)

        
        self.test = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=False)

        self.all = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers, 
                shuffle=False,
                pin_memory=False)


class IntersectionSampler(object):
    """
    Sampler for only pulling examples shared by all classes
    """

    def __init__(self, dataset, valid_dataset, companion_datasets, val_split=0.1,
            test_split=0.5, shuffle=True, num_workers=8,
            batch_size=64, random_seed=73, transform=None):
        
        if isinstance(valid_dataset, list):
            valid_dataset = ConcatDataset(*valid_dataset)
        
        if isinstance(dataset, list):
            dataset = ConcatDataset(*dataset)
        
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.val_split = val_split
        self.test_split = test_split

        names = [dataset.name]
        if dataset.sub_name is not None:
            names.append(dataset.sub_name)
        for d in companion_datasets:
            names.append(d.name)
            if d.sub_name is not None:
                names.append(d.sub_name)

        logging.info("Names")
        logging.info(names)
        split_fn = get_split_filename(names, val_split,
                test_split, random_seed, "intersection")

        logging.info(split_fn)
        if os.path.exists(split_fn):
            with open(split_fn, "r") as f:
                splits = json.load(f)

            self.splits = splits
            indices = []
            indices.extend(self.splits["train"])
            indices.extend(self.splits["val"])
            indices.extend(self.splits["test"])
        else:
            
            if hasattr(dataset, "get_base_classes_ids"):

                base_classes = dataset.get_base_classes_ids()
                logging.info(base_classes)

                for ds in companion_datasets:
                    base_classes = base_classes.intersection(ds.get_base_classes_ids())

                logging.info("Intersected Base Classes: %s" % str(base_classes))
                indices = []

                for cl in base_classes:
                    if hasattr(dataset, "base2instance"):
                        cl_transform = dataset.base2instance()[cl]
                    else:
                        cl_transform = cl
                    #logging.info(cl_transform)
                    #logging.info(cl)
                    #logging.info(type(cl_transform))
                    #cl_transform = torch.Tensor(cl_transform)
                    #logging.info(cl_transform)
                    #logging.info(dataset.class_samples)
                    indices.extend(dataset.class_samples[cl_transform])

            else:
                indices = list(range(len(dataset)))


            #logging.info("Intersection Indices: %s" % str(indices))

            train_ids, val_ids, test_ids = split(indices, self.val_split,
                    self.test_split, random_seed)
            
            splits = {"train": train_ids, "val": val_ids, "test": test_ids}
            
            self.splits = splits
            with open(split_fn, "w") as f:
                json.dump(splits, f)

        self.train = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["train"]),
                pin_memory=False)


        self.val = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["val"]),
                pin_memory=False)

        
        self.test = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["test"]),
                pin_memory=False)

        self.all = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers, 
                sampler=SubsetRandomSampler(indices),
                pin_memory=False)



def split(indices, val_split, test_split, random_seed):
    if test_split + val_split >= 1.0:
        raise ValueError("Val and test split greater than 1.0")

    dataset_size = len(indices)
    val_split_cnt = int(np.floor(val_split * dataset_size))
    test_split_cnt = int(np.floor(test_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices) 

    test_indices = indices[:test_split_cnt]
    val_indices = indices[test_split_cnt:test_split_cnt+val_split_cnt]
    train_indices = indices[test_split_cnt + val_split_cnt:]
    
    return train_indices, val_indices, test_indices



class ContextConcatDataset(torch.utils.data.Dataset):

    def __init__(self, context, *datasets):


        # Raise if context and datset lengths aren't identical
        if context.size(0) != len(datasets):
            print(datasets)
            raise ValueError("Length of context: %d is not equal to len datasets: %d" % (len(context), len(datasets)))

        self.datasets = datasets
        self.context = context #torch.tensor(context, dtype=torch.float32)


        self.root = "_".join([os.path.basename(ds.root) for ds in datasets]) + "_context"
        
        self.name = "_".join([ds.name for ds in datasets]) + "_grouped"

        sub_names = [ds.sub_name for ds in datasets if ds.sub_name is not None]
        if len(sub_names) > 0:
            self.sub_name = "_".join(sub_names) + "_grouped"
        else:
            self.sub_name = None
        
        self.lens = []

        for i, ds in enumerate(datasets):
            #if i == 0:
            if i == 0:
                self.lens.append(len(ds))
            else:
                self.lens.append(self.lens[-1] + len(ds))

        logging.info("Len Context dataset: %d" % self.__len__())


    def __getitem__(self, i):

        for ds_idx, ds_len in enumerate(self.lens):
            if i < ds_len:
                if ds_idx == 0: 
                    shift = 0
                else:
                    shift = self.lens[ds_idx - 1]

                inputs, target =  self.datasets[ds_idx][i - shift]
                ctx = self.context[ds_idx]
                return inputs, ctx, target
        
        raise ValueError("Index out of range on combined datasets")

    def __len__(self):
        return self.lens[-1]


class ConcatDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

        self.root = "_".join([os.path.basename(ds.root) for ds in datasets]) + "_grouped"

        self.name = "_".join([ds.name for ds in datasets]) + "_grouped"

        sub_names = [ds.sub_name for ds in datasets if ds.sub_name is not None]
        if len(sub_names) > 0:
            self.sub_name = "_".join(sub_names) + "_grouped"
        else:
            self.sub_name = None

        self.lens = []

        for i, ds in enumerate(datasets):
            if i == 0:
                self.lens.append(len(ds))
            else:
                self.lens.append(self.lens[-1] + len(ds))

    def __getitem__(self, i):

        for ds_idx, ds_len in enumerate(self.lens):
            if i < ds_len:
                if ds_idx == 0: 
                    shift = 0
                else:
                    shift = self.lens[ds_idx - 1]

                return self.datasets[ds_idx][i - shift]
        
        raise ValueError("Index out of range on combined datasets")

    def __len__(self):
        return self.lens[-1]



class StandardSampler(object):
    """
    Sampler for Train/Val/Test splits
    """

    def __init__(self, dataset, valid_dataset, companion_datsets, val_split=0.1, test_split=0.5, shuffle=True, num_workers=8, batch_size=64, random_seed=73, transform=None):

        if isinstance(dataset, list):
            dataset = ConcatDataset(*dataset)

        if isinstance(valid_dataset, list):
            valid_dataset = ConcatDataset(*valid_dataset)
        
        self.dataset = dataset
        self.valid_dataset = valid_dataset

        self.val_split = val_split
        self.test_split = test_split

        names = [dataset.name]
        if dataset.sub_name is not None:
            names.append(dataset.sub_name)
        
        split_fn = get_split_filename(names, 
                val_split, test_split, random_seed, "standard")

        if os.path.exists(split_fn):
            try:
                with open(split_fn, "r") as f:
                    splits = json.load(f)
            except Exception as e:
                print(split_fn)
                raise ValueError(e)

            self.splits = splits
        else:
            indices = list(range(len(dataset)))
            train_ids, val_ids, test_ids = split(indices, self.val_split,
                    self.test_split, random_seed)

            splits = {"train": train_ids, "val": val_ids, "test": test_ids}
            
            self.splits = splits
            with open(split_fn, "w") as f:
                json.dump(splits, f)


        self.train = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["train"]),
                pin_memory=False)


        self.val = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["val"]),
                pin_memory=False)

        
        self.test = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers,
                sampler=SubsetRandomSampler(self.splits["test"]),
                pin_memory=False)

        self.all = DataLoader(self.valid_dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=shuffle,
                pin_memory=False)










class Splitter(object):
    """
    Splitter: Helper class to split single file datasets into train and test
    """
    def __init__(self, dataset_cls, data_file, validation_split=0.2, shuffle=True, num_workers=4, batch_size=32, random_seed=43,
            transform=None, dedup=False):
        print("Initializer has been kicked off")
        
        dataset = dataset_cls(data_file, transform=transform)
        
        self.dataset = dataset

        train_indices, val_indices = self.split(dataset, validation_split, random_seed)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(dataset, batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=train_sampler, 
                                        pin_memory=False)

        self.valid_loader = DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers,
                                        sampler=valid_sampler,
                                        pin_memory=False)
        
        self.disc_train_1 = self.train_loader
        self.disc_train_2 =  self.valid_loader
        self.disc_valid_1 = self.train_loader
        self.disc_valid_2 = self.valid_loader
        print("Initialization has finished")

    def split(self, indices, validation_split, random_seed, downsize=None):
        start = time.time()
        dataset_size = len(indices)
        split = int(np.floor(validation_split * dataset_size))
        train_size = int(np.floor((1-validation_split) * dataset_size))

        np.random.seed(random_seed)
        np.random.shuffle(indices)
        if downsize:
            print("Downsizing: %d" %train_size)
            train_size =  int(np.floor(downsize * train_size))
            print("Downsized: %d" %train_size)
            

        train_indices, val_indices = indices[split:split+train_size], indices[:split]
        end = time.time()
        print("split time laoders: %f" %(end-start))
        return train_indices, val_indices



class DoubleSplitter(Splitter):
    """
    Splitter: Helper class to split multiple file datasets into test/train
              as well as discriminator test/train
    """

    def __init__(self, dataset_cls, df_1, df_2, validation_split=0.2, shuffle=True, 
                 num_workers=4, batch_size=32, random_seed=43, transform=None,
                 downsize=None, d1_map=None, d2_map=None, dedup=False):

        print("Initializer has been kicked off")
        dataset_1 = dataset_cls(df_1, transform=transform)
        dataset_2 = dataset_cls(df_2, transform=transform)
        if dedup:
            d1_indices, d2_indices = dedup_datasets(df_1, df_2, dataset_cls)
        else:
            d1_indices = list(range(len(dataset_1)))
            d2_indices = list(range(len(dataset_2)))

        self.dataset = dataset_1
        if d1_map:
            train_indices_1 = d1_map["train_indices"]
            val_indices_1 = d1_map["val_indices"]
        else:
            print("Down Sizing")
            train_indices_1, val_indices_1 = self.split(d1_indices, validation_split, 
                    random_seed, downsize=downsize)

        if d2_map:
            train_indices_2 = d2_map["train_indices"]
            val_indices_2 = d2_map["val_indices"]
        else:
            print("Down Sizing")
            train_indices_2, val_indices_2 = self.split(d2_indices, validation_split, 
                    random_seed, downsize=downsize)


        self.train_indices_1 = train_indices_1
        self.train_indices_2 = train_indices_2
        self.val_indices_1 = val_indices_1
        self.val_indices_2 = val_indices_2

        train_sampler_1 = SubsetRandomSampler(train_indices_1)
        valid_sampler_1 = SubsetRandomSampler(val_indices_1)
        
        train_sampler_2 = SubsetRandomSampler(train_indices_2)
        valid_sampler_2 = SubsetRandomSampler(val_indices_2)

        self.disc_train_1 = DataLoader(dataset_1, batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=train_sampler_1,
                                        pin_memory=False)

        self.disc_train_2 = DataLoader(dataset_2, batch_size=batch_size,
                                         num_workers=num_workers,
                                        sampler=train_sampler_2,
                                        pin_memory=False)

        self.disc_valid_1 = DataLoader(dataset_1, batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=valid_sampler_1,
                                        pin_memory=False)

        self.disc_valid_2 = DataLoader(dataset_2, batch_size=batch_size,
                                         num_workers=num_workers,
                                        sampler=valid_sampler_2,
                                        pin_memory=False)


        if downsize:
            dataset_size = len(dataset_1)
            indices = list(range(dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            print("Downsize start: :%d" %dataset_size)
            dataset_size = int(np.floor(dataset_size * downsize))
            print("Downsize end: :%d" %dataset_size)
            indices = indices[:dataset_size]
            train_sampler = SubsetRandomSampler(indices)
            self.train_loader = DataLoader(dataset_1, batch_size=batch_size,
                                            num_workers=num_workers,
                                            sampler=train_sampler, 
                                            pin_memory=False)
        else:
            self.train_loader = DataLoader(dataset_1, batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=True, pin_memory=False)

        self.valid_loader = DataLoader(dataset_2, batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True, pin_memory=False)

        print("Initialization has finished")


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

