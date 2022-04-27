import os
import sys
import time
import argparse
import loaders
import tempfile
import shutil
import json

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import logging
import logging.handlers

from importlib import import_module

import utils as d_utils
import networks

import torchvision
from torchvision import datasets as tvdatasets
from torchvision import transforms
from torchvision import utils
from torchvision import models
import torch
import torch.nn as nn

import logging

from algorithms.discriminator import *
from algorithms.pretext import *
from algorithms.distances import *
from algorithms.temperature import *

from functools import partial

import tempfile

matplotlib.use('Agg')

# Select Datasets to be used
    
    # Must Load train/validation splits for multiple ds
    #[imagenetc-1, imagenetc-2, imagenet1k-1, imagenet-all]

# Select Models and/or Model Architectures to be used
    
    # Loads Model Architectures and Specified Weights
    # Most should be pretraining, but can load view an initializer as well

# Perform Task Specific Model Training (Using train/val only)
    # [skip for imagenet] as this is already done but maybe do some domaingen shit

# Perform Domain-Specific Calibration/prediction (Using train/val/test for (train domains))
    # Do source evaluation for base case
    # Do temperature scaling for second base case
    # Do Disc Accuracy based model performance prediction for this case
    # Create a wrapper model for domain-specific calibration

# Evaluate Performances on Held-Out Domains (Calculate metrics for held out domains (test only))
    # Predict Accuracy, Accuracy Gap, and Percentage Accuracy Gap
    # Calculate ECE and Brier Scores for true metrics in this case

# Additionally train and superset model which uses results from multiple models

def build_arg_parser():
    parser = argparse.ArgumentParser("Give config for running performance prediction")
    parser.add_argument("--cal-alg", dest="cal_alg", required=True)
    parser.add_argument("--base-alg", dest="base_alg", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--batch-norm", dest="batch_norm", action='store_true')
    parser.add_argument("--cuda", default=0, type=int, help="number of cuda device to use")
    parser.add_argument("--bootstrap", default=1000, type=int, help="number of runs")
    return parser


class MyDataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0, name=None):
        super(MyDataParallel, self).__init__(module, device_ids, output_device, dim)

        self.dims = self.module.dims
        self.net = self.module.net

    def disc_forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.disc_forward(*inputs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("Module must have its parameters and buffers"
                                   "on device {} (device_ids[0]) but found one of"
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.disc_forward(*inputs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        replicas = [r.disc_forward for r in replicas]
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    """
    def __getattr__(self, name):
        return getattr(self.module, name)
    """

def plot_results(preds, targets, plot_fn, mse, mae, r2):
    sns.set_style("whitegrid")
    sns.regplot(preds, targets)
    plt.xlabel("Predicted Accuracy Drop")
    plt.ylabel("Actual Accuracy Drop") 
    plt.title("MSE: %f\t MAE: %f\t R2: %f" %(mse, mae, r2))
    plt.savefig(plot_fn)


def loadDatasets(dataset_params):
    file_locs = json.load(open("datasets.json"))

    dataset_struct = {}
    for ds_type, dataset_names in dataset_params.items():
        datasets = []

        for (ds_name, data_transform, valid_transform, sub_name) in dataset_names:
            ds_file_loc = file_locs[ds_name]
            if sub_name is not None:
                ds_file_loc = os.path.join(ds_file_loc, sub_name)
            
            if "imagenetc"  in ds_name:
                dataset = d_utils.ImgCDataset(ds_file_loc, transform=data_transform, 
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImgCDataset(ds_file_loc, transform=valid_transform, 
                        name=ds_name, sub_name=sub_name)
            elif "imagenetv2" in ds_name:
                dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=data_transform, 
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet-a" in ds_name:
                dataset = d_utils.ImageNetADataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetADataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet-r" in ds_name:
                dataset = d_utils.ImageNetRDataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetRDataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet-vid" in ds_name:
                dataset = d_utils.ImageNetVidRobustDataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetVidRobustDataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet" == ds_name:
                dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet-train" == ds_name:
                dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=valid_transform,
                        name=ds_name, sub_name=sub_name)
            elif "imagenet-sketch" == ds_name:
                dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.ImageNetV2Dataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
            else:
                dataset = d_utils.StandardDataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.StandardDataset(ds_file_loc, transform=valid_transform, 
                        name=ds_name, sub_name=sub_name)

            datasets.append((dataset, val_dataset))

        dataset_struct[ds_type] = datasets

    #logging.info(dataset_struct)
    return dataset_struct

def modelLoader(model_params):
    model = networks.FeatureWrapper(model_params["net"],
            layer_probe=model_params["layer_probe"],
            pretrained=model_params["pretrained"])

    return model


def trainAlgorithm(base_train_params, device=None):

    return EmptyTrainer()


def calibrationAlgorithm(cal_algorithm, device=None, model_naming=None, conn=None):

    if "binary_discriminator" in cal_algorithm["name"]:
        return BinaryDiscriminator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
    elif "multiclass_discriminator" in cal_algorithm["name"]:
        return MultiClassDiscriminator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
        raise NotImplementedError()
        #return MultiClassDiscriminator(cal_algorithm.parameters)
    elif "pretext" in cal_algorithm["name"]:
        return PretextCalibrator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
    elif "frechet" in cal_algorithm["name"]:
        return FrechetModelDistance(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
    elif "mmd" in cal_algorithm["name"]:
        return MaximumMeanDiscrepancy(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
    elif "temperature" in cal_algorithm["name"]:
        return TemperatureScalingAlg(cal_algorithm["parameters"], device=device,
                model_naming=model_naming, conn=conn)
    else:
        raise NotImplementedError()

    return None


def samplerGenerator(sampler_params):

    if "intersection" in sampler_params["name"]:
    #if "intersection" in sampler_params:
        sampler = partial(loaders.IntersectionSampler, **sampler_params["sampler_params"])
    elif "standard" in sampler_params["name"]:
        sampler = partial(loaders.StandardSampler, **sampler_params["sampler_params"])
    else:
        raise NotImplementedError

    return sampler


def buildConfig(config_fn):
    filename = os.path.abspath(os.path.expanduser(config_fn))

    fileExtname = os.path.splitext(filename)[1]
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir,
                suffix=fileExtname)



        temp_config_name = os.path.basename(temp_config_file.name)

        shutil.copyfile(filename, temp_config_file.name)

        # only do .py for now
        if filename.endswith('.py'):
            temp_module_name = os.path.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)

            mod = import_module(temp_module_name)
            sys.path.pop(0)

            cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                    }

            del sys.modules[temp_module_name]
        else:
            raise ValueError("bad filename extension")

        temp_config_file.close()


    return cfg_dict["config"]
    #return cfg_dict["config"]




def classifier_preprocess(model, dataset, device=None, filename=None):

    outputs_list = []
    targets_list = []

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(dataset)):

            inputs = inputs.to(device)
            outputs = model(inputs)

            outputs = outputs.cpu()
            outputs_list.append(outputs)
            targets_list.append(targets)

        o_stack = torch.cat(outputs_list, dim=0)
        t_stack = torch.cat(targets_list, dim=0)
        t_stack = t_stack.type(torch.LongTensor)

    #tf = tempfile.TemporaryFile(dir=directory)
    return d_utils.BootstrapStoreDataset(o_stack, t_stack, name=dataset.dataset.name,
        class_sublist=dataset.dataset.get_base_classes_ids(), 
        sub_name=dataset.dataset.sub_name,
        root=dataset.dataset.root,
        dataset=dataset.dataset,
        filename=filename,
        save=True)


def features_preprocess(model, dataset, device=None, filename=None):
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(dataset)):

            inputs = inputs.to(device)
            outputs = model.disc_forward(inputs)

            outputs = outputs.cpu()
            outputs_list.append(outputs)
            targets_list.append(targets)

        o_stack = torch.cat(outputs_list, dim=0)
        t_stack = torch.cat(targets_list, dim=0)
        t_stack = t_stack.type(torch.LongTensor)

    #tf = tempfile.TemporaryFile(dir=directory)
    return d_utils.BootstrapStoreDataset(o_stack, t_stack, name=dataset.dataset.name,
        class_sublist=dataset.dataset.get_base_classes_ids(),
        sub_name=dataset.dataset.sub_name,
        root=dataset.dataset.root,
        dataset=dataset.dataset,
        filename=filename,
        save=True)



def preprocess_data(model, sampler_gen, base_ds, cal_ds,
            test_ds, cal_alg, device=None, cls_eval_params=None,
            directory=None, token=None, conn=None):
    # Make this the pseudosampler
    # Do real sampler in main
    # Temperatures turned off

    pre_base_ds = []
    pre_cal_ds = []
    pre_test_ds = []
    if "temperature" in cal_alg["name"]:
        # Classification features

        for base_data in base_ds:
            base_pre_token = token_maker(token, base_data)

            if d_utils.db.check_existence("bootstrap", {"id": base_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset - Base")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": base_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")
                
                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)


                base_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=base_data[0].name,
                        class_sublist=base_data[0].get_base_classes_ids(), 
                        sub_name=base_data[0].sub_name,
                        root=base_data[0].root,
                        dataset=base_data[0],
                        filename=fn)

                pre_base_ds.append(base_pre_output)
            else:
                base_ds_sampler = sampler_gen(base_data[0], base_data[1])

                fn = os.path.join(directory, base_pre_token + "_data")

                base_pre_output = classifier_preprocess(model, base_ds_sampler.all, 
                                            device=device, filename=fn) #directory=directory)

                boot_de = {"id": base_pre_token, "filename": fn, 
                        "dataset": base_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_base_ds.append(base_pre_output)

        for target_data in cal_ds:
            target_pre_token = token_maker(token, target_data)
            
            if d_utils.db.check_existence("bootstrap", {"id": target_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset-Cal")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": target_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")

                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)

                target_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=target_data[0].name,
                        class_sublist=target_data[0].get_base_classes_ids(), 
                        sub_name=target_data[0].sub_name,
                        root=target_data[0].root,
                        dataset=target_data[0],
                        filename=fn)

                pre_cal_ds.append(target_pre_output)
            else:

                fn = os.path.join(directory, target_pre_token + "_data")
                target_ds_sampler = sampler_gen(target_data[0], target_data[1])

                target_pre_output = classifier_preprocess(model, target_ds_sampler.all,
                                            device=device, filename=fn)
                
                boot_de = {"id": target_pre_token, "filename": fn, 
                        "dataset": target_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_cal_ds.append(target_pre_output)

            
        for target_data in test_ds:
            target_pre_token = token_maker(token, target_data)

            if d_utils.db.check_existence("bootstrap", {"id": target_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset - Test")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": target_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")

                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)

                target_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=target_data[0].name,
                        class_sublist=target_data[0].get_base_classes_ids(), 
                        sub_name=target_data[0].sub_name,
                        root=target_data[0].root,
                        dataset=target_data[0],
                        filename=fn)

                pre_test_ds.append(target_pre_output)
            else:

                fn = os.path.join(directory, target_pre_token + "_data")
                target_ds_sampler = sampler_gen(target_data[0], target_data[1])

                target_pre_output = classifier_preprocess(model, target_ds_sampler.all,
                                            device=device, filename=fn)
                
                boot_de = {"id": target_pre_token, "filename": fn, 
                        "dataset": target_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_test_ds.append(target_pre_output)


    else:
        # FEATURES
        for base_data in base_ds:
            base_pre_token = token_maker(token, base_data, "feat")
            
            if d_utils.db.check_existence("bootstrap", {"id": base_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset - Base")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": base_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")
                
                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)


                base_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=base_data[0].name,
                        class_sublist=base_data[0].get_base_classes_ids(), 
                        sub_name=base_data[0].sub_name,
                        root=base_data[0].root,
                        dataset=base_data[0],
                        filename=fn)

                pre_base_ds.append(base_pre_output)
            else:
                base_ds_sampler = sampler_gen(base_data[0], base_data[1])

                fn = os.path.join(directory, base_pre_token + "_data")

                base_pre_output = features_preprocess(model, base_ds_sampler.all, 
                                            device=device, filename=fn) #directory=directory)

                boot_de = {"id": base_pre_token, "filename": fn, 
                        "dataset": base_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_base_ds.append(base_pre_output)
            

        for target_data in cal_ds:
            target_pre_token = token_maker(token, target_data, "feat")
            
            if d_utils.db.check_existence("bootstrap", {"id": target_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset-Cal")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": target_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")

                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)

                target_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=target_data[0].name,
                        class_sublist=target_data[0].get_base_classes_ids(), 
                        sub_name=target_data[0].sub_name,
                        root=target_data[0].root,
                        dataset=target_data[0],
                        filename=fn)

                pre_cal_ds.append(target_pre_output)
            else:

                fn = os.path.join(directory, target_pre_token + "_data")
                target_ds_sampler = sampler_gen(target_data[0], target_data[1])

                target_pre_output = features_preprocess(model, target_ds_sampler.all,
                                            device=device, filename=fn)
                
                boot_de = {"id": target_pre_token, "filename": fn, 
                        "dataset": target_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_cal_ds.append(target_pre_output)

            
        for target_data in test_ds:
            target_pre_token = token_maker(token, target_data, "feat")
            
            if d_utils.db.check_existence("bootstrap", {"id": target_pre_token}, conn=conn):
                logging.info("Reading Bootstrap Dataset - Test")
                boot_de = d_utils.db.get_selection_table("bootstrap",
                        match_e={"id": target_pre_token},
                        conn=conn)
                boot_de = boot_de[0]

                fn = boot_de["filename"]
                
                fp = np.load(fn + ".npy")

                y_fp = np.load(fn + "_y.npy")

                x = torch.from_numpy(fp)
                y = torch.from_numpy(y_fp).type(torch.LongTensor)

                target_pre_output = d_utils.BootstrapStoreDataset(x, y, 
                        name=target_data[0].name,
                        class_sublist=target_data[0].get_base_classes_ids(), 
                        sub_name=target_data[0].sub_name,
                        root=target_data[0].root,
                        dataset=target_data[0],
                        filename=fn)

                pre_test_ds.append(target_pre_output)
            else:

                fn = os.path.join(directory, target_pre_token + "_data")
                target_ds_sampler = sampler_gen(target_data[0], target_data[1])

                target_pre_output = features_preprocess(model, target_ds_sampler.all,
                                            device=device, filename=fn)
                
                boot_de = {"id": target_pre_token, "filename": fn, 
                        "dataset": target_data[0].name}

                check = d_utils.db.insert_into_table("bootstrap", boot_de, conn=conn)
                pre_test_ds.append(target_pre_output)


    return pre_base_ds, pre_cal_ds, pre_test_ds




def build_bootstrap(pre_base_datasets,
                pre_cal_datasets,
                pre_test_datasets, directory=None):

    new_base_datasets = []
    new_cal_datasets = []
    new_test_datasets = []
    
    for base_data in pre_base_datasets:
        b_len = len(base_data)
        indices = list(range(b_len))

        bs_indices = np.random.choice(indices, size=b_len, replace=True)

        data = [base_data[idx][0] for idx in bs_indices]
        labels = [base_data[idx][1] for idx in bs_indices]
        
        o_stack = torch.stack(data, dim=0)
        t_stack = torch.LongTensor(labels)
        t_stack = t_stack.type(torch.LongTensor)
        tf = tempfile.NamedTemporaryFile(dir=directory) #.name
        boot_ds = d_utils.BootstrapStoreDataset(o_stack, t_stack, name=base_data.name,
                class_sublist=base_data.class_sublist,
                sub_name=base_data.sub_name,
                dataset=base_data,
                root=base_data.root,
                filename=tf)

        new_base_datasets.append((boot_ds, boot_ds))


    for cal_data in pre_cal_datasets:
        c_len = len(cal_data)
        indices = list(range(c_len))

        bs_indices = np.random.choice(indices, size=c_len, replace=True)

        data = [cal_data[idx][0] for idx in bs_indices]
        labels = [cal_data[idx][1] for idx in bs_indices]
        
        o_stack = torch.stack(data, dim=0)
        t_stack = torch.LongTensor(labels)
        t_stack = t_stack.type(torch.LongTensor)
        tf = tempfile.NamedTemporaryFile(dir=directory)#.name
        boot_ds = d_utils.BootstrapStoreDataset(o_stack, t_stack, name=cal_data.name,
                class_sublist=cal_data.class_sublist,
                dataset=cal_data,
                sub_name=cal_data.sub_name,
                root=base_data.root,
                filename=tf)

        new_cal_datasets.append((boot_ds, boot_ds))

    for test_data in pre_test_datasets:
        t_len = len(test_data)
        indices = list(range(t_len))

        bs_indices = np.random.choice(indices, size=t_len, replace=True)

        data = [test_data[idx][0] for idx in bs_indices]
        labels = [test_data[idx][1] for idx in bs_indices]
        
        o_stack = torch.stack(data, dim=0)
        t_stack = torch.LongTensor(labels)
        t_stack = t_stack.type(torch.LongTensor)
        #t_stack = torch.LongTensor(labels)
        tf = tempfile.NamedTemporaryFile(dir=directory)#.name
        boot_ds = d_utils.BootstrapStoreDataset(o_stack, t_stack, name=test_data.name,
                class_sublist=test_data.class_sublist,
                sub_name=test_data.sub_name,
                dataset=test_data,
                root=test_data.root,
                filename=tf)

        new_test_datasets.append((boot_ds, boot_ds))
    
    return new_base_datasets, new_cal_datasets, new_test_datasets


def main(args):
    
    
    config = {
            "cal_algorithm": buildConfig(args.cal_alg),
            "model": buildConfig(args.model),
            "base_train_algorithm": buildConfig(args.base_alg),
            "dataset": buildConfig(args.dataset),
            "sampler": buildConfig(args.sampler)
            }

    db_location = "pd_data/bootstrap_central.db"

    bootstrap_conn = d_utils.db.create_connection(db_location)
    d_utils.db.create_table(bootstrap_conn, d_utils.db.create_dist_results)
    d_utils.db.create_table(bootstrap_conn, d_utils.db.create_cls_results)
    d_utils.db.create_table(bootstrap_conn, d_utils.db.create_perf_results)
    d_utils.db.create_table(bootstrap_conn, d_utils.db.create_bootstrap_results)

    dt = datetime.now()
    dt_string = dt.isoformat().replace(":", "_").replace(".", "_")

    base_naming =  os.path.basename(args.cal_alg).replace(".py", "") + "_" + \
            os.path.basename(args.base_alg).replace(".py", "")+ "_" + \
            os.path.basename(args.dataset).replace(".py", "")  + "_" + \
            os.path.basename(args.sampler).replace(".py", "") + "_" + \
            os.path.basename(args.model).replace(".py", "")

    if args.batch_norm:
        base_naming = base_naming + "_" + "batch_norm"


    filename = os.path.join("pp_logging", base_naming  + "_" + dt_string)

    results_dir_name = os.path.join("results", base_naming + "_" + dt_string)
    os.mkdir(results_dir_name)

    should_roll_over  = os.path.isfile(filename)

    handler = logging.handlers.RotatingFileHandler(filename, mode="w+", backupCount=2)
    if should_roll_over:
        handler.doRollover()

    logging.basicConfig(
            level=logging.DEBUG,
            handlers=[handler, logging.StreamHandler()],
            format="%(message)s")


    logging.error(config)
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

    datasets = loadDatasets(config["dataset"])
    model = networks.FeatureWrapper(config["model"]["net"],
            layer_probe=config["model"]["layer_probe"],
            pretrained=config["model"]["pretrained"])
    
    if config["model"]["net"].startswith("alexnet") or config["model"]["net"].startswith("vgg"):
        model.net.features = torch.nn.DataParallel(model.net.features)
        model.to(device)
    else:
        model = MyDataParallel(model)
        model.to(device)

    
    bootstrap = args.bootstrap 
    model_naming = "tmp_bootstrap_%d_%s_%s_%s" %(args.bootstrap, 
            config["model"]["net"], 
            config["model"]["layer_probe"],
            config["model"]["pretrained"])
    
    logging.info(model_naming)
    #modelLoader(config.model)
    bt_algorithm = trainAlgorithm(config["base_train_algorithm"], device=device)

    cpu_device = torch.device("cpu")
    #cal_algorithm = calibrationAlgorithm(config["cal_algorithm"], device=cpu_device,
    #        model_naming=model_naming, conn=bootstrap_conn)
    
    cal_algorithm = calibrationAlgorithm(config["cal_algorithm"], device=device,
            model_naming=model_naming, conn=bootstrap_conn)
    
    # real sampler gen
    sampler_config = config["sampler"].copy()
    sampler_params = sampler_config["sampler_params"].copy()
    #sampler_params["batch_size"] = 10000
    sampler_config["sampler_params"] = sampler_params
    #sampler_config["sampler_params"]["batch_size"] = 10000
    sampler_gen = samplerGenerator(config["sampler"])
    bs_sampler_gen = samplerGenerator(sampler_config)

    logging.info("About to start training")
    
    #commence training

    # Skip Optimization
    model, train_results = bt_algorithm.train(model, datasets["base"], sampler_gen) 


    train_results_fn = os.path.join(results_dir_name, "train_results.json")
    with open(train_results_fn, "w+") as f:
        json.dump(train_results, f, indent=1)

    logging.info("Task-Specific Training Complete")

    # add option for turning on eval
    if not args.batch_norm:
        logging.info("Batch norm is off")
        model.eval()
    else:
        logging.info("Adaptive Batch Norm is running")


    #pseudo_sampler = loaders.PseudoSampler()
    pseudo_sampler = partial(loaders.PseudoSampler, **config["sampler"]["sampler_params"])

    logging.info("PreProcessing Started")

    with tempfile.TemporaryDirectory() as td:

        preprocess_data_dir = "preprocess_data"
        if not os.path.isdir(preprocess_data_dir):
            os.mkdir(preprocess_data_dir)


        if "temperature" in config["cal_algorithm"]["name"]:
            layer_probe = "preds"
        else:
            layer_probe = config["model"]["layer_probe"]
        
        pre_token = token_maker(config["model"]["net"], layer_probe,
                config["model"]["pretrained"],
                #base_datasets, 
                config["base_train_algorithm"], pseudo_sampler, 
                args.batch_norm)

        pre_base_datasets, pre_cal_datasets, pre_test_datasets = preprocess_data(model, pseudo_sampler, 
                datasets["base"],
                datasets["cal"],
                datasets["test"],
                config["cal_algorithm"], 
                device=device,
                directory=preprocess_data_dir,
                token=pre_token,
                conn=bootstrap_conn)
        
        logging.info("PreProcessing Completed")

        if "temperature" in config["cal_algorithm"]["name"] :
            fake_net = MyDataParallel(networks.IdentityNet(dims=1000))
            fake_net.to(device)
        elif config["model"]["net"].startswith("alexnet") or config["model"]["net"].startswith("vgg"):
            f_net = networks.BootstrapNet(model.class_features, dims=model.dims)
            f_net.to(device)
            fake_net = MyDataParallel(f_net)
            fake_net.to(device)
        else:
            # Write code to extract classifier from this function
            f_net = networks.BootstrapNet(model.module.class_features, dims=model.module.dims)
            f_net.to(device)
            fake_net = MyDataParallel(f_net)
            fake_net.to(device)
            #networks.IdentityNet(dims=model.dims)
            #raise NotImplementedError()

        """
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        logging.info("Free memory :%f" %f)
        1/0
        """
        tot_s_time = time.time()


        for sample_idx in range(args.bootstrap):
            s_time = time.time()
            base_datasets, cal_datasets, test_datasets = build_bootstrap(pre_base_datasets,
                    pre_cal_datasets,
                    []) #pre_test_datasets)
            
            # Create Unique Tokenizer for run
            base_token = token_maker(config["model"]["net"], config["model"]["layer_probe"],
                    config["model"]["pretrained"],
                    base_datasets, config["base_train_algorithm"], sampler_gen, 
                    args.batch_norm)
           
            logging.info(base_datasets)
            logging.info(cal_datasets)
            cal_results = cal_algorithm.train(fake_net, base_datasets, cal_datasets, 
                    bs_sampler_gen, base_token) 

            logging.info("Calibrated Model Trained")
            
            cal_token = token_maker(datasets["cal"])

            for c in  cal_datasets:
                del c

            _, _, test_datasets = build_bootstrap([],
                    [], #pre_cal_datasets,
                    pre_test_datasets)
            
            preds = []
            gaps =  []
            for test_ds in test_datasets: # datasets["test"]:
                logging.info("Dataset Root: %s"% test_ds[0].root)
                pred, gap, evals, final_eval = cal_algorithm.group_prediction(test_ds, 
                        bs_sampler_gen, base_token, cal_token)

                preds.append(pred) #.cpu().item())
                gaps.append(gap) #.cpu().item())
                logging.info("Pred: %f" % pred)
                logging.info("Gap: %f" % gap)
                logging.info(evals)
                logging.info(final_eval)

            preds_arr = np.stack(preds, axis=0)
            gaps_arr  = np.stack(gaps, axis=0)

            for b in base_datasets:
                del b
            #del cal_datasets.tensors
            for t in  test_datasets:
                del t


            from sklearn.metrics import mean_squared_error
            logging.info(gaps_arr)
            logging.info(preds_arr)
            test_mse = mean_squared_error(gaps_arr, preds_arr)

            logging.info("Test MSE: %f" % test_mse)

            from sklearn.metrics import mean_absolute_error
            test_mae = mean_absolute_error(gaps_arr, preds_arr)

            logging.info("Test MAE: %f" % test_mae)

            from sklearn.metrics import r2_score

            test_r2_score = r2_score(gaps_arr, preds_arr)

            logging.info("Test R2 Score: %f" % test_r2_score)

            max_error = np.max(np.abs(preds_arr - gaps_arr), 0)

            logging.info("Max Error: %f" %max_error)

            e_time = time.time()
            logging.info("Iteration: %d\t run_time: %f\t tot_time: %f" %
                    (sample_idx, e_time - s_time, e_time - tot_s_time))



if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)




