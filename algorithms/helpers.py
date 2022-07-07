import torch

import torch
import torch.nn as nn
import numpy as np
import logging

import tempfile
import shutil
import sys
from importlib import import_module

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .evaluators import *
from loaders import *
import utils as d_utils

import pandas as pd
import os
import collections
import xxhash



from itertools import chain
from tqdm import tqdm

import wandb


from loaders import ContextConcatDataset
import loaders

from functools import partial


dist_dataframe_fn = "pd_data/dist_df.pkl"
cls_dataframe_fn = "pd_data/cls_df.pkl"
model_dataframe_fn = "pd_data/model_df.pkl"
perf_dataframe_fn = "pd_data/performance.pkl"


df_dir = os.path.dirname(dist_dataframe_fn)

if not os.path.exists(df_dir):
    os.mkdir(df_dir)

global DIST_DF
if not os.path.exists(dist_dataframe_fn):
    DIST_DF = pd.DataFrame(columns=["accuracy", "auc", "brier", "ece",
        "test_avg", "loss", "score", "mse", "mae"])
else:
    DIST_DF = pd.read_pickle(dist_dataframe_fn)

global CLS_DF
if not os.path.exists(cls_dataframe_fn):
    CLS_DF = pd.DataFrame(columns=["accuracy", "auc", "brier", "ece",
        "test_avg", "loss"])
else:
    CLS_DF = pd.read_pickle(cls_dataframe_fn)

global MODEL_DF
if not os.path.exists(model_dataframe_fn):
    MODEL_DF = pd.DataFrame(columns=["model_loc"])
else:
    MODEL_DF = pd.read_pickle(model_dataframe_fn)

global PERF_DF
if not os.path.exists(perf_dataframe_fn):
    PERF_DF = pd.DataFrame(columns=["base_ds", "target_ds", "model",
        "token", "cal_alg", "gap","pred_gap",  "error", 
        "accuracy", "pred_acc"])
else:
    PERF_DF = pd.read_pickle(perf_dataframe_fn)


def temperature_token_builder(base_token, t_ds, sampler_gen, 
        cls_eval_params, aux_ds, filt=None, conv=None,
        temperature=1.0):

    if (filt is not None or conv is not None) and temperature == 1.0:
        return token_maker(base_token, t_ds, sampler_gen, 
                cls_eval_params, aux_ds)
    elif filt is not None or conv is not None:
        return token_maker(base_token, t_ds, sampler_gen,
                cls_eval_params, aux_ds, temperature)
    elif temperature == 1.0:
        return token_maker(base_token, t_ds, sampler_gen,
                cls_eval_params)
    else:
        return token_maker(base_token, t_ds, sampler_gen,
                cls_eval_params, temperature)



def check_token_ds(token, df):
    logging.info("checking token: %s" % token)
    match =  token in df.index
    logging.info("Found: %r" %match)
    return match


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


def update_df(results, token, df, df_filename):
    """Updating Database with results"""

    df = pd.read_pickle(df_filename)
    logging.info(df.index)
    logging.info("Initial DF Len: %d" %(len(df)))
    for k, v in results.items():
        logging.info("token: %s    %s: %s" % (token, k, str(v)))
        df.loc[token, k] = v

    logging.info("Final DF Len: %d" %(len(df)))
    logging.info(df.index)
    df.to_pickle(df_filename)

    DIST_DF = pd.read_pickle(dist_dataframe_fn)
    CLS_DF = pd.read_pickle(cls_dataframe_fn)
    PERF_DF = pd.read_pickle(perf_dataframe_fn)

    return df




def get_filters_and_conversions(base_ds, target_ds):

    if hasattr(target_ds, "base_filter"):
        base_filter = target_ds.base_filter()
    else:
        base_filter = None

    if hasattr(target_ds, "target_filter"):
        target_filter = target_ds.target_filter()
    else:
        target_filter = None

    if hasattr(target_ds, "base_conversion"):
        base_conversion = target_ds.base_conversion()
    else:
        base_conversion = None

    if hasattr(target_ds, "target_conversion"):
        target_conversion = target_ds.target_conversion()
    else:
        target_conversion = None

    return base_filter, base_conversion, target_filter, target_conversion


def samplerGenerator(sampler_params):

    if "intersection" in sampler_params["name"]:
    #if "intersection" in sampler_params:
        sampler = partial(loaders.IntersectionSampler, **sampler_params["sampler_params"])
    elif "standard" in sampler_params["name"]:
        sampler = partial(loaders.StandardSampler, **sampler_params["sampler_params"])
    else:
        raise NotImplementedError

    return sampler


def loadDatasets(dataset_params, ds_file="datasets.json"):
    file_locs = json.load(open(ds_file))

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



def build_optimizer(net, train_params=None, discriminator=False):
     
    logging.info("Learning Rate: %f" % train_params["lr"])


    if discriminator is False:
        print("just net")
        params = net.parameters()

    elif train_params["opt_wt"] == "full_opt":
        print("full net")
        params = [
                  {"params": net.feat_net.parameters()},
                  {"params": net.disc_net.parameters()}
                 ]
    elif train_params["opt_wt"] == "graded_opt":
        print("graded net")
        params = [
                  {"params": net.disc_net.parameters()},
                  {"params": net.feat_net.parameters(), "lr": 
                      train_params["lr"] * train_params["graded_discount"]}
                 ]
    elif train_params["opt_wt"] == "disc_opt":
        print("disc net")
        params = net.disc_net.parameters()
    else:
        raise NotImplementedError

    if train_params["opt"] == "sgd":
        optimizer = optim.SGD(params, lr=train_params["lr"], 
                momentum=train_params["momentum"], 
                weight_decay=train_params["wd"])
    elif train_params["opt"] == "adam":
        optimizer = optim.Adam(params, lr=train_params["lr"], 
                weight_decay=train_params["wd"])
    elif train_params["opt"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=train_params["lr"], 
                momentum=train_params["momentum"],
                weight_decay=train_params["wd"])

    return optimizer



def train_disc_model(disc_net, ds_iter, train_params, device=None, build_labels=None,
        base_ds_name="base", target_ds_name="target", batch_norm=False):
    """
    Revized version of train model for simpler saving and loading
    """
    epochs = train_params["epochs"]
    criterion = train_params["criterion"]
    scheduler_params = train_params["scheduler"]

    optimizer = build_optimizer(disc_net,
            train_params=train_params["optimizer"],
            discriminator=True)


    scheduler = StepLR(optimizer,
            step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"])

    cnt = 0
    running_loss = 0.0
    best_acc = 0.0
    valid_loss = 0.0
    acc = 0.0

    ds_train = [ds.train for ds in ds_iter]
    ds_val = [ds.val for ds in ds_iter]

    for epoch in tqdm(range(epochs)):

        if batch_norm:
            #disc_net.train()
            disc_net.feat_net.train()
            disc_net.disc_net.train()
        else:
            #disc_net.eval()
            disc_net.feat_net.eval()
            disc_net.disc_net.eval()

        for i, train_samples  in enumerate(zip(*ds_train)):
            cnt += 1

            if build_labels:
                inputs, labels = build_labels(train_samples)
            else:
                inputs, labels = build_flatten_labels(train_samples)


            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = disc_net(inputs)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if cnt % 20 == 19:
                wandb.log({
                    "%s/%s" %(base_ds_name, target_ds_name) : {
                        "loss": running_loss / 20,
                        "epoch": epoch}
                    }
                    )

                logging.info("[%d, %5d] loss: %.3f" %(epoch + 1, 
                    cnt + 1,  running_loss / 20))
                running_loss = 0.0



        valid_loss = 0.0
        acc = 0.0

        # Validation Step
        logging.info("Epoch: %d" % epoch)
        acc_calc = AccuracyCalculator()
        loss_calc = LossAggregator(criterion)

        
        disc_net.feat_net.eval()
        disc_net.disc_net.eval()
        #disc_net.eval()

        with torch.no_grad():
            for i, val_samples in enumerate(zip(*ds_val)):

                if build_labels:
                    test_inputs, targets = build_labels(val_samples)
                else:
                    test_inputs, targets = build_flatten_labels(val_samples)
                

                test_inputs = test_inputs.to(device)
                targets = targets.to(device)

                outputs = disc_net(test_inputs)
                acc_calc.update(outputs, targets)
                loss_calc.update(outputs, targets)

            acc = acc_calc.results()
            a_loss = loss_calc.results()

            logging.info("Test Balance Acc: %f\t Loss: %f" % (acc, a_loss))

            wandb.log({
                    "%s Test Accuracy" % (target_ds_name) : acc,
                    "%s Test Loss" % (target_ds_name) : a_loss
                })

           
            if epoch % 5 == 0:
                acc_calc = AccuracyCalculator()
                loss_calc = LossAggregator(criterion)
                
                for i, train_samples  in enumerate(zip(*ds_train)):

                    if build_labels:
                        inputs, labels = build_labels(train_samples)
                    else:
                        inputs, labels = build_flatten_labels(train_samples)

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = disc_net(inputs)
                    
                    acc_calc.update(outputs, labels)
                    loss_calc.update(outputs, labels)
                
                acc = acc_calc.results()
                a_loss = loss_calc.results()

                logging.info("Train Balance Acc: %f\t Loss: %f" % (acc, a_loss))

                wandb.log({
                        "%s Train Accuracy" % (target_ds_name) : acc,
                        "%s Train Loss" % (target_ds_name) : a_loss
                    })


        scheduler.step()




    logging.info("Model finished training")

    return disc_net


def train_model(feat_net, disc_net, ds_iter, train_params, device=None, build_labels=None,
        base_ds_name="base", target_ds_name="target", batch_norm=False):
    epochs = train_params["epochs"]
    criterion = train_params["criterion"]
    scheduler_params = train_params["scheduler"]

    optimizer = build_optimizer(feat_net,
            disc_net=disc_net,
            train_params=train_params["optimizer"])


    scheduler = StepLR(optimizer,
            step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"])

    cnt = 0
    running_loss = 0.0
    best_acc = 0.0
    valid_loss = 0.0
    acc = 0.0

    ds_train = [ds.train for ds in ds_iter]
    ds_val = [ds.val for ds in ds_iter]

    def disc_pass(x):
        x = feat_net.disc_forward(x)
        #logging.info("Feat Shape Helper: %s" %(str(x.shape)))
        #x = x.detach()
        x = disc_net(x)
        return x

    for epoch in tqdm(range(epochs)):

        if batch_norm:
            feat_net.train()
            disc_net.train()
        else:
            feat_net.eval()
            disc_net.eval()

        for i, train_samples  in enumerate(zip(*ds_train)):
            cnt += 1

            if build_labels:
                inputs, labels = build_labels(train_samples)
            else:
                inputs, labels = build_flatten_labels(train_samples)


            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = disc_pass(inputs)
            
            #logging.info(outputs)
            #logging.info(labels)


            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if cnt % 20 == 19:
                wandb.log({
                    "%s/%s" %(base_ds_name, target_ds_name) : {
                        "loss": running_loss / 20,
                        "epoch": epoch}
                    }
                    )

                logging.info("[%d, %5d] loss: %.3f" %(epoch + 1, 
                    cnt + 1,  running_loss / 20))
                running_loss = 0.0



        valid_loss = 0.0
        acc = 0.0

        # Validation Step
        logging.info("Epoch: %d" % epoch)
        acc_calc = AccuracyCalculator()
        loss_calc = LossAggregator(criterion)

        feat_net.eval()
        disc_net.eval()
        with torch.no_grad():
            for i, val_samples in enumerate(zip(*ds_val)):

                if build_labels:
                    test_inputs, targets = build_labels(val_samples)
                else:
                    test_inputs, targets = build_flatten_labels(val_samples)
                

                test_inputs = test_inputs.to(device)
                targets = targets.to(device)

                outputs = disc_pass(test_inputs)
                acc_calc.update(outputs, targets)
                loss_calc.update(outputs, targets)

            acc = acc_calc.results()
            a_loss = loss_calc.results()

            logging.info("Test Balance Acc: %f\t Loss: %f" % (acc, a_loss))

            wandb.log({
                    "%s Test Accuracy" % (target_ds_name) : acc,
                    "%s Test Loss" % (target_ds_name) : a_loss
                })

           
            if epoch % 5 == 0:
                acc_calc = AccuracyCalculator()
                loss_calc = LossAggregator(criterion)
                
                for i, train_samples  in enumerate(zip(*ds_train)):

                    if build_labels:
                        inputs, labels = build_labels(train_samples)
                    else:
                        inputs, labels = build_flatten_labels(train_samples)

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = disc_pass(inputs)
                    
                    acc_calc.update(outputs, labels)
                    loss_calc.update(outputs, labels)
                
                acc = acc_calc.results()
                a_loss = loss_calc.results()

                logging.info("Train Balance Acc: %f\t Loss: %f" % (acc, a_loss))

                wandb.log({
                        "%s Train Accuracy" % (target_ds_name) : acc,
                        "%s Train Loss" % (target_ds_name) : a_loss
                    })


        scheduler.step()




    logging.info("Model finished training")

    return feat_net, disc_net


def build_features(cls_evals, disc_evals, regressor_features):
    features = torch.zeros((len(disc_evals), len(regressor_features)), dtype=torch.float32)

    logging.info(features.shape)

    for i, e in enumerate(disc_evals):
        for j, r_feat in enumerate(regressor_features):
            if r_feat == "a_proxy":
                features[i][j] = 2*(1- 2*(1-e["accuracy"]))
            elif r_feat == "test_avg":
                features[i][j] = cls_evals[i][1][r_feat]
            elif "diff" in r_feat:
                r_feat_root = r_feat.replace("_diff", "")
                features[i][j] =  cls_evals[i][0][r_feat_root] - cls_evals[i][1][r_feat_root]
            elif r_feat == "base_acc":
                features[i][j] = cls_evals[i][0]["accuracy"]
            elif r_feat == "entropy":
                logging.info(cls_evals[i][1])
                features[i][j] = cls_evals[i][1][r_feat]
            else:
                features[i][j] = e[r_feat]

    return features


def build_single_feature(cls_eval, disc_eval, regressor_features):

    feature = torch.zeros((len(regressor_features)), dtype=torch.float32)

    for j, r_feat in enumerate(regressor_features):
        if r_feat == "a_proxy":
            feature[j] = 2*(1 - 2*(1 - disc_eval["accuracy"]))
        elif r_feat == "test_avg":
            feature[j] = cls_eval[1][r_feat]
        #elif r_feat == "test_avg_diff":
        elif r_feat == "base_acc":
            feature[j] = cls_eval[0]["accuracy"]
        elif "diff" in r_feat: #r_feat == "test_avg_diff":
            r_feat_root = r_feat.replace("_diff", "")
            feature[j] =  cls_eval[0][r_feat_root] - cls_eval[1][r_feat_root]
        elif r_feat == "entropy":
            feature[j] = cls_eval[1][r_feat]
        else:
            feature[j] = disc_eval[r_feat]

    return feature


def train_dynamic_temperature(regressor_model, feat_net, train_ds, 
        cls_evals, disc_evals,
        regressor_features, regressor_target, regressor_params, 
        regressor_eval_params, sampler_gen, device=None):

    #context = []

    regressor_model.to(device)
    
    context = build_features(cls_evals, disc_evals, regressor_features)
    
    logging.info("context")
    logging.info(str(context))

    train_ds_0 = [d[0] for d in train_ds]
    train_ds_1 = [d[1] for d in train_ds]
    ctx_ds = ContextConcatDataset(context, *train_ds_0)
    ctx_ds_2 = ContextConcatDataset(context, *train_ds_1)

    epochs =  regressor_params["epochs"]
    criterion = regressor_params["criterion"]
    scheduler_params = regressor_params["scheduler"]
    optimizer_params = regressor_params["optimizer"]

    lr = optimizer_params["lr"]


    logging.info("Dynamic temperature starting")

    cnt = 0
    running_loss = 0.0

    optimizer = build_optimizer(regressor_model, 
            train_params=optimizer_params)

    target_ds_sampler = sampler_gen(ctx_ds, ctx_ds_2, [])

    bf, bc, targetf, tc = get_filters_and_conversions(ctx_ds,
                                         ctx_ds_2)

    logits_list = []
    ctx_list = []
    labels_list = []

    if len(list(regressor_model.parameters())) <= 2:
        logging.info("Staring LBFGS parameter setup")
        with torch.no_grad():


            for inputs, ctx, targets in tqdm(target_ds_sampler.train):

                if targetf is not None:
                    inputs, targets = targetf(inputs, targets)

                inputs = inputs.to(device)

                logits = feat_net(inputs)
                if hasattr(feat_net.net, "combine_preds"):
                    logits = feat_net.net.combine_preds(logits)


                if tc is not None:
                    logits, targets = tc(logits, targets)

                logits = logits.cpu()

                logits_list.append(logits)
                ctx_list.append(ctx)
                labels_list.append(targets)

            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            ctx = torch.cat(ctx_list)
        
        logging.info("logits Size: %d, %d" % (logits.size(0), logits.size(1)))
        
        regressor_model.to("cpu")
        
        logging.info("Weight: %f\t Bias: %f" % (regressor_model.net.weight.item(),
                 regressor_model.net.bias.item()))
        pre_loss = nn.functional.cross_entropy(logits, labels).item()
        logging.info("Before Optimization NLL: %f" % pre_loss)
        logging.info(str(ctx[0:30]))
        optimizer = optim.LBFGS([regressor_model.net.weight, 
                                 regressor_model.net.bias], lr=lr, max_iter=500)

        def eval_ctx():
            scaled_logits = logits/regressor_model(ctx)
            loss = nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        logging.info("Optimizer for LBFGS started")
        optimizer.step(eval_ctx)
        
        temps = regressor_model(ctx)
        scaled_logits = logits/temps
        post_loss = nn.functional.cross_entropy(scaled_logits, labels).item()
        logging.info("After Optimization NLL: %f" % post_loss)
        regressor_model.to(device)
    else:
        raise ValueError("We haven implemented conversion for this method of training")
        logging.info("Optimized via SGD for non-linear model")
        for ep in tqdm(range(epochs)):

            for inputs, ctx, targets in target_ds_sampler.train:

                inputs, ctx, targets = inputs.to(device), ctx.to(device), targets.to(device)

                optimizer.zero_grad()

                logits = feat_net(inputs)
                logits.detach()

                temperature = regressor_model(ctx)

                logits = logits / temperature

                loss = nn.functional.cross_entropy(logits, targets)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if cnt % 500 == 499:
                    logging.info("[%d, %d] loss: %.3f" %(ep, cnt, running_loss/500))
                    logging.info("Weight: %.3f\tbias: %.3f" % (regressor_model.net.weight.item(), 
                        regressor_model.net.bias.item()))
                    running_loss = 0.0

                cnt += 1
    
    mae = MAE_Score()
    mse = MSE_Score()
    r2_score =  R2_Score()
    worst_error = WorstError()

    pred_results = []
    gt_results = []

    logging.info("Weight: %f\t Bias: %f" % (regressor_model.net.weight.item(),
                 regressor_model.net.bias.item()))
    with torch.no_grad():
        for ds_idx, target_ds in enumerate(train_ds):

            pred_accuracy = AvgMaxScore(raw_scores=True)

            target_ds_sampler = sampler_gen(target_ds[0], target_ds[1], [])

            for inputs, targets in tqdm(target_ds_sampler.all):
                
                if targetf is not None:
                    inputs, targets = targetf(inputs, targets)
                
                inputs = inputs.to(device)
                logits = feat_net(inputs)
                
                if hasattr(feat_net.net, "combine_preds"):
                    logits = feat_net.net.combine_preds(logits)

                if tc is not None:
                    logits, targets = tc(logits, targets)
                
                ctx = context[ds_idx].to(device)
                temperature = regressor_model(ctx)
                logits = logits / temperature

                logits.cpu()

                pred_accuracy.update(logits, targets)

            p_acc = pred_accuracy.results()#.item()
            p_acc = p_acc.unsqueeze(0)
            pred_gap = cls_evals[ds_idx][0][regressor_target] - p_acc
            pred_gap = pred_gap#.unsqueeze(0)
            real_gap = cls_evals[ds_idx][0][regressor_target] - cls_evals[ds_idx][1][regressor_target]
            real_gap = torch.Tensor([real_gap])#.unsqueeze(0)

            
            logging.info("Predicted Accuracy: %f\t Real Accuracy: %f" % (p_acc, cls_evals[ds_idx][1][regressor_target]))
            logging.info("Predicted Gap: %f\t Real Gap: %f" % (pred_gap, real_gap))

            pred_gap = pred_gap.cpu()
            real_gap = real_gap.cpu()

            pred_results.append(pred_gap.item())
            gt_results.append(real_gap.item())

            
            logging.info("GAPS")
            logging.info(pred_gap)
            logging.info(real_gap)
            mae.update(pred_gap, real_gap)
            mse.update(pred_gap, real_gap)
            r2_score.update(pred_gap, real_gap)
            worst_error.update(pred_gap, real_gap)


        mae_val = mae.results().item()
        mse_val = mse.results().item()
        r2_val  = r2_score.results().item()
        we_val = worst_error.results()

        logging.info("MAE: %f" % mae_val)
        logging.info("MSE: %f" % mse_val)
        logging.info("R2 Score: %f" % r2_val)
        logging.info("Worst Error: %f" % we_val)
        logging.info(pred_results)
        logging.info(gt_results)

    
    #reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val}
    reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val,
            "preds": pred_results, "targets": gt_results, "worst_error": we_val}
    return regressor_model, reg_results


def fixed_accuracy_feat_diff(regressor_model, 
        cls_evals, disc_evals,
        regressor_features, regressor_target, regressor_params, 
        regressor_eval_params, sampler_gen, device=None):



    mae = MAE_Score()
    mse = MSE_Score()
    r2_score =  R2_Score()
    worst_error = WorstError()
    pred_results = []
    gt_results = []
    
    with torch.no_grad():
        for c_idx, ce in enumerate(cls_evals):
            for reg_feat in regressor_features:
                if reg_feat == "base_acc":
                    pred_gap = ce[0]["accuracy"] - ce[0]["accuracy"]
                else:
                    pred_gap = ce[0][reg_feat] - ce[1][reg_feat]

            true_gap = ce[0][regressor_target] - ce[1][regressor_target]

            true_gap = true_gap#.unsqueeze(0)
            pred_gap = pred_gap#.unsqueeze(0)

            pred_results.append(pred_gap)#.item())
            gt_results.append(true_gap)#.item())
            logging.info(pred_gap)
            logging.info(true_gap)
            tg = torch.Tensor([true_gap])
            pg = torch.Tensor([pred_gap])

            mse.update(pg, tg)
            mae.update(pg, tg)
            r2_score.update(pg, tg)
            worst_error.update(pg, tg)
            #mse.update(pred_gap, true_gap)
            #mae.update(pred_gap, true_gap)
            #r2_score.update(pred_gap, true_gap)

        
        mae_val = mae.results().item()
        mse_val = mse.results().item()
        r2_val  = r2_score.results().item()
        we_val  = worst_error.results()

    logging.info("MAE: %f" % mae_val)
    logging.info("MSE: %f" % mse_val)
    logging.info("R2 Score: %f" % r2_val)
    logging.info("Worst Error: %f" % we_val)

    reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val,
            "worst_error": we_val,
            "preds": pred_results, "targets": gt_results}
    
    return regressor_model, reg_results

def fixed_accuracy_raw(regressor_model, 
        cls_evals, disc_evals,
        regressor_features, regressor_target, regressor_params, 
        regressor_eval_params, sampler_gen, device=None):



    mae = MAE_Score()
    mse = MSE_Score()
    r2_score =  R2_Score()
    worst_error = WorstError()
    pred_results = []
    gt_results = []
    
    with torch.no_grad():
        for c_idx, ce in enumerate(cls_evals):
            for reg_feat in regressor_features:
                if reg_feat == "base_acc":
                    pred_gap = ce[0]["accuracy"] - ce[0]["accuracy"]
                else:
                    pred_gap = ce[0]["accuracy"] - ce[1][reg_feat]

            true_gap = ce[0][regressor_target] - ce[1][regressor_target]

            true_gap = true_gap#.unsqueeze(0)
            pred_gap = pred_gap#.unsqueeze(0)

            pred_results.append(pred_gap)#.item())
            gt_results.append(true_gap)#.item())
            logging.info(pred_gap)
            logging.info(true_gap)
            tg = torch.Tensor([true_gap])
            pg = torch.Tensor([pred_gap])

            mse.update(pg, tg)
            mae.update(pg, tg)
            r2_score.update(pg, tg)
            worst_error.update(pg, tg)
            #mse.update(pred_gap, true_gap)
            #mae.update(pred_gap, true_gap)
            #r2_score.update(pred_gap, true_gap)

        
        mae_val = mae.results().item()
        mse_val = mse.results().item()
        r2_val  = r2_score.results().item()
        we_val  = worst_error.results()

    logging.info("MAE: %f" % mae_val)
    logging.info("MSE: %f" % mse_val)
    logging.info("R2 Score: %f" % r2_val)
    logging.info("Worst Error: %f" % we_val)

    reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val,
            "worst_error": we_val,
            "preds": pred_results, "targets": gt_results}
    
    return regressor_model, reg_results
    


def train_regressor(regressor_model, class_evals, disc_evals, 
        regressor_features, regressor_target, regressor_params, 
        regressor_eval_params, device=None):

    labels =  torch.zeros((len(class_evals), 1), dtype=torch.float32)

    ### ASSUMES BINARY
    for i, ev in enumerate(class_evals):
        labels[i] = ev[0][regressor_target] - ev[1][regressor_target]
    
    #torch.cat(tuple([e[regressor_target] for e in evals]), 0)

    features = build_features(class_evals, disc_evals, regressor_features)
     
    logging.info(features)
    logging.info(labels)
    
    # Do with SKLEARN for now??
    epochs = regressor_params["epochs"]
    criterion = regressor_params["criterion"]
    scheduler_params = regressor_params["scheduler"]
    optimizer_params = regressor_params["optimizer"]

    logging.info("Original Learning Rate: %f" % optimizer_params["lr"])
    # Implement Build optimizer
    optimizer = build_optimizer(regressor_model, 
            train_params=optimizer_params)

    scheduler = StepLR(optimizer, step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"])

    ds = TensorDataset(features, labels)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    cnt  = 0

    running_loss = 0.0

    regressor_model.to(device)


    # Check If Linear
    if len(list(regressor_model.parameters())) <= 2:
        logging.info("Found a linear model so using Least Squares Solver")

        bias_vec = torch.ones((len(class_evals),1), dtype=torch.float32)
        joint_features = torch.cat((features, bias_vec), 1)
        R, Q = torch.lstsq(labels, joint_features)
        
        regressor_model.net.weight = torch.nn.Parameter(R[0:features.size(1)])
        regressor_model.net.bias = torch.nn.Parameter(R[features.size(1)])
        regressor_model.to(device)
    else:

        for ep in tqdm(range(epochs)):

            for i, (x,y) in enumerate(dl):

                cnt += 1

                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)
                
                outputs = regressor_model(x)

                
                loss = nn.functional.mse_loss(outputs, y)
                

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                if cnt % 200 == 199:
                    
                    # logging and writer in future version
                    logging.info("[%d, %5d] loss: %.3f" % (ep + 1, cnt + 1,  running_loss / 200))
                    running_loss = 0.0


    mae = MAE_Score()
    mse = MSE_Score()
    r2_score =  R2_Score()
    worst_error = WorstError()
    pred_results = []
    gt_results = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dl):

            x = x.to(device)
            outputs = regressor_model(x)

            outputs = outputs.cpu()

            pred_results.append(outputs)#.item())
            gt_results.append(y)#.item())
            
            mae.update(outputs, y)
            mse.update(outputs, y)
            r2_score.update(outputs, y)
            worst_error.update(outputs, y)


        mae_val = mae.results().item()
        mse_val = mse.results().item()
        r2_val  = r2_score.results().item()
        we_val  = worst_error.results()

        logging.info("MAE: %f" % mae_val)
        logging.info("MSE: %f" % mse_val)
        logging.info("R2 Score: %f" % r2_val)
        logging.info("Worst Error: %f" %we_val)

    pred_results = torch.cat(pred_results, 0).tolist()
    gt_results = torch.cat(gt_results, 0).tolist()


    reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val,
            "worst_error": we_val,
            "preds": pred_results, "targets": gt_results}
    
    return regressor_model, reg_results



class LogPredModel(nn.Module):


    def __init__(self, net):
        super(LogPredModel, self).__init__()
        #self.net = nn.Sequential(net, nn.Sigmoid())
        self.net = net

    def forward(self, x):
        #return self.net(1/x)
        x = self.net(x)
        x = x / torch.sqrt( 1 + torch.pow(x, 2))
        x = (x +1)/2
        return x


def train_logistic_regressor(regressor_model, class_evals, disc_evals, 
        regressor_features, regressor_target, regressor_params, 
        regressor_eval_params, device=None):

    
    regressor_model = LogPredModel(regressor_model.net)

    labels =  torch.zeros((len(class_evals), 1), dtype=torch.float32)

    ### ASSUMES BINARY
    for i, ev in enumerate(class_evals):
        #labels[i] = ev[0][regressor_target] - ev[1][regressor_target]
        labels[i] = ev[1][regressor_target]
    

    features = build_features(class_evals, disc_evals, regressor_features)
     
    logging.info(features)
    logging.info(labels)
    
    # Do with SKLEARN for now??
    epochs = regressor_params["epochs"]
    criterion = regressor_params["criterion"]
    scheduler_params = regressor_params["scheduler"]
    optimizer_params = regressor_params["optimizer"]

    logging.info("Original Learning Rate: %f" % optimizer_params["lr"])
    # Implement Build optimizer
    optimizer = build_optimizer(regressor_model, 
            train_params=optimizer_params)

    scheduler = StepLR(optimizer, step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"])

    ds = TensorDataset(features, labels)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    cnt  = 0

    running_loss = 0.0

    regressor_model.to(device)


    # Check If Linear

    for ep in tqdm(range(epochs)):

        for i, (x,y) in enumerate(dl):

            cnt += 1

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)
            
            outputs = regressor_model(x)
            
            loss = nn.functional.mse_loss(outputs, y)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if cnt % 200 == 199:
                
                # logging and writer in future version
                logging.info("[%d, %5d] loss: %.3f" % (ep + 1, cnt + 1,  running_loss / 200))
                running_loss = 0.0


    mae = MAE_Score()
    mse = MSE_Score()
    r2_score =  R2_Score()
    worst_error = WorstError()
    pred_results = []
    gt_results = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dl):

            x = x.to(device)
            outputs = regressor_model(x)

            outputs = outputs.cpu()

            pred_results.append(outputs)#.item())
            gt_results.append(y)#.item())
            
            mae.update(outputs, y)
            mse.update(outputs, y)
            r2_score.update(outputs, y)
            worst_error.update(outputs, y)


        mae_val = mae.results().item()
        mse_val = mse.results().item()
        r2_val  = r2_score.results().item()
        we_val  = worst_error.results()

        logging.info("MAE: %f" % mae_val)
        logging.info("MSE: %f" % mse_val)
        logging.info("R2 Score: %f" % r2_val)
        logging.info("Worst Error: %f" %we_val)

    pred_results = torch.cat(pred_results, 0).tolist()
    gt_results = torch.cat(gt_results, 0).tolist()


    reg_results = {"mae": mae_val, "mse": mse_val, "r2_score": r2_val,
            "worst_error": we_val,
            "preds": pred_results, "targets": gt_results}
    
    return regressor_model, reg_results

