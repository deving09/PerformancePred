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
from algorithms.vanilla import *

import wandb

from functools import partial


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
    parser.add_argument("--bootstraps", default=None, type=int, help="number of runs")
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
            elif "cifar10" == ds_name:
                dataset = d_utils.CIFAR10(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name, train=True)
                val_dataset = d_utils.CIFAR10(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name, train=True)
            elif "cifar10c" in ds_name:
                dataset = d_utils.CIFAR10_C(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.CIFAR10_C(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
            elif ("cifar10.1-v4" == ds_name) or ("cifar10.1-v6" == ds_name):
                dataset = d_utils.CIFAR10_1(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name) #, train=True)
                val_dataset = d_utils.CIFAR10_1(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name) #, train=True)
            elif "cifar100" == ds_name:
                dataset = d_utils.CIFAR100(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name, train=True)
                val_dataset = d_utils.CIFAR100(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name, train=False)
            elif "cifar100c" in ds_name:
                dataset = d_utils.CIFAR100_C(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)#, train=True)
                val_dataset = d_utils.CIFAR100_C(root=ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name) #, train=False)
            else:
                dataset = d_utils.StandardDataset(ds_file_loc, transform=data_transform,
                        name=ds_name, sub_name=sub_name)
                val_dataset = d_utils.StandardDataset(ds_file_loc, transform=valid_transform, 
                        name=ds_name, sub_name=sub_name)

            datasets.append((dataset, val_dataset))

        dataset_struct[ds_type] = datasets

    return dataset_struct

def modelLoader(model_params):
    model = networks._load_base_model(model_params["net"],
            layer_probe=model_params["layer_probe"],
            pretrained=model_params["pretrained"])

    return model


def trainAlgorithm(base_train_params, device=None):

    if "empty" in base_train_params["name"]:
        return EmptyTrainer()
    elif "vanilla" in base_train_params["name"]:
        return VanillaTrain(base_train_params["parameters"], device=device)
    elif "posthoc" in base_train_params["name"]:
        return PosthocTrain(base_train_params["parameters"], device=device)
    else:
        raise ValueError("Base train algorithms not specified")

    return None

def calibrationAlgorithm(cal_algorithm, device=None, model_naming=None):

    if "binary_discriminator" in cal_algorithm["name"]:
        logging.info("Binary Discriminator Loading")
        return BinaryDiscriminator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
    elif "multiclass_discriminator" in cal_algorithm["name"]:
        return MultiClassDiscriminator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
        raise NotImplementedError()
    elif "pretext" in cal_algorithm["name"]:
        return PretextCalibrator(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
    elif "frechet" in cal_algorithm["name"]:
        return FrechetModelDistance(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
    elif "mmd" in cal_algorithm["name"]:
        return MaximumMeanDiscrepancy(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
    elif "temperature" in cal_algorithm["name"]:
        return TemperatureScalingAlg(cal_algorithm["parameters"], device=device,
                model_naming=model_naming)
    else:
        raise NotImplementedError()

    return None


def samplerGenerator(sampler_params):

    if "intersection" in sampler_params["name"]:
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


def main(args):
    
    
    config = {
            "cal_algorithm": buildConfig(args.cal_alg),
            "model": buildConfig(args.model),
            "base_train_algorithm": buildConfig(args.base_alg),
            "dataset": buildConfig(args.dataset),
            "sampler": buildConfig(args.sampler)
            }


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

    wandb.init(project="Discriminators", entity="deving09", config=config)

    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")


    datasets = loadDatasets(config["dataset"])

    model = networks._load_base_model(config["model"]["net"],
            layer_probe=config["model"]["layer_probe"],
            pretrained=config["model"]["pretrained"],
            num_classes=len(datasets["base"][0][0].get_base_classes_ids()))
    
    if config["model"]["net"].startswith("alexnet") or config["model"]["net"].startswith("vgg"):
        logging.info("Old School Data Parallel")
        model.net.features = torch.nn.DataParallel(model.net.features)
        model.to(device)
    else:
        logging.info("New School Data Parallel")
        model = MyDataParallel(model)
        model.to(device)


    model_naming = "%s_%s_%s" %(config["model"]["net"], 
            config["model"]["layer_probe"],
            config["model"]["pretrained"])
    
    logging.info(model_naming)
    bt_algorithm = trainAlgorithm(config["base_train_algorithm"], device=device)
    cal_algorithm = calibrationAlgorithm(config["cal_algorithm"], device=device,
            model_naming=model_naming)

    sampler_gen = samplerGenerator(config["sampler"])

    logging.info("About to start training")
    
    
    # add option for turning on eval
    if not args.batch_norm:
        logging.info("Batch norm is off")
        model.eval()
    else:
        logging.info("Adaptive Batch Norm is running")

    # Skip Optimization
    model, train_results = bt_algorithm.train(model, datasets["base"], datasets["base"], sampler_gen, batch_norm=args.batch_norm) 


    train_results_fn = os.path.join(results_dir_name, "train_results.json")
    with open(train_results_fn, "w+") as f:
        json.dump(train_results, f, indent=1)

    logging.info("Task-Specific Training Complete")


    # Create Unique Tokenizer for run
    base_token = token_maker(config["model"]["net"], config["model"]["layer_probe"],
            config["model"]["pretrained"],
            datasets["base"], config["base_train_algorithm"], sampler_gen, args.batch_norm)


    logging.info("Base Datasets: %s" % str(datasets["base"]))
    
    cal_results = cal_algorithm.train(model, datasets["base"], datasets["cal"], sampler_gen, base_token, batch_norm=args.batch_norm) 

    # params=config.cal_params)
    
    cal_results_fn = os.path.join(results_dir_name, "cal_results.json")
    with open(cal_results_fn, "w+") as f:
        json.dump(cal_results, f, indent=1)

    cal_plot_fn = os.path.join(results_dir_name, "cal_plot.png")
    plot_results(cal_results["preds"], 
                 cal_results["targets"], 
                 cal_plot_fn, 
                 cal_results["mse"], 
                 cal_results["mae"], 
                 cal_results["r2_score"])

    logging.info("Calibrated Model Trained")

    logging.info(datasets["cal"])
    cal_token = token_maker(datasets["cal"])
    logging.info("cal token: %s" % cal_token)

    preds = []
    gaps =  []
    for test_ds in datasets["test"]:
        logging.info("Dataset Root: %s"% test_ds[0].root)
        pred, gap, evals, final_eval = cal_algorithm.group_prediction(test_ds, 
                sampler_gen, base_token, cal_token)

        preds.append(pred) #.cpu().item())
        gaps.append(gap) #.cpu().item())
        logging.info("Pred: %f" % pred)
        logging.info("Gap: %f" % gap)
        logging.info(evals)
        logging.info(final_eval)
        logging.info("Base Token: %s\t Cal Token: %s" %(base_token, cal_token))
    #run_tests(calibrated_model, datasets["test"])

    preds_arr = np.stack(preds, axis=0)
    gaps_arr  = np.stack(gaps, axis=0)

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

    # Update pandas dataframe or pkl or CSV
    # Write Update

    test_results = {"mae": test_mae,
            "mse": test_mse,
            "r2_score": test_r2_score,
            "preds": preds,
            "targets": gaps,
            "max_error": max_error}

    test_results_fn = os.path.join(results_dir_name, "test_results.json")
    with open(test_results_fn, "w+") as f:
        json.dump(test_results, f, indent=1)

    test_plot_fn = os.path.join(results_dir_name, "test_plot.png")
    plot_results(test_results["preds"], 
                 test_results["targets"], 
                 test_plot_fn, 
                 test_results["mse"], 
                 test_results["mae"], 
                 test_results["r2_score"])



if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)




