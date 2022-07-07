import os
import sys
import argparse
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import sigmoid

import loaders
#from networks import build_model
import networks
import random
import time
import datetime
import logging
import json

import torchvision
from torchvision import datasets as tvdatasets
from torchvision import transforms
from torchvision import utils
from torchvision import models

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import utils as disc_utils
from algorithms import helpers
from algorithms.evaluators import *

def build_arg_parser():
    parser = argparse.ArgumentParser("Train discriminator between domains")
    parser.add_argument("--source", required=True, help="source dataset", 
            choices=["imagenet", "imagenetv2-threshold", "imagenetv2-matched", 
                "imagenetv2-top", "imagenet-vid", "imagenet-a", "imagenetv2-matched-fixed",
                "imagenetc-blur", "imagenetc-extra", "imagenetc-noise", "imagenetc-digital",
                "imagenetc-motion-blur", "imagenetc-defocus-blur", "imagenetc-glass-blur",
                "imagenetc-zoom-blur", "imagenetc-gaussian-noise","imagenetc-impulse-noise",
                "imagenetc-shot-noise","imagenetc-digital-jpeg","imagenetc-digital-elastic",
                "imagenetc-digital-contrast","imagenetc-digital-pixelate","imagenetc-extra-gaussian",
                "imagenetc-extra-saturate", "imagenetc-extra-speckle","imagenetc-extra-spatter",
                "imagenet-a", "imagenet-o", "imagenetv2-threshold-fixed", "imagenetv2-top-fixed",
                "imagenetc-weather-frost"])
    parser.add_argument("--target", required=True, help="target dataset", 
            choices=["imagenet", "imagenetv2-threshold", "imagenetv2-matched", 
                "imagenetv2-top", "imagenet-vid", "imagenet-a", "imagenetv2-matched-fixed",
                "imagenetc-blur", "imagenetc-extra", "imagenetc-noise", "imagenetc-digital",
                "imagenetc-motion-blur", "imagenetc-defocus-blur", "imagenetc-glass-blur",
                "imagenetc-zoom-blur", "imagenetc-gaussian-noise","imagenetc-impulse-noise",
                "imagenetc-shot-noise","imagenetc-digital-jpeg","imagenetc-digital-elastic",
                "imagenetc-digital-contrast","imagenetc-digital-pixelate","imagenetc-extra-gaussian",
                "imagenetc-extra-saturate", "imagenetc-extra-speckle","imagenetc-extra-spatter",
                "imagenet-a", "imagenet-o", "imagenetv2-threshold-fixed", "imagenetv2-top-fixed",
                "imagenetc-weather-frost"])
    
    parser.add_argument("--source_sub", default=None, help="subdirectory for source")
    parser.add_argument("--target_sub", default=None, help="subdirectory for target")
    parser.add_argument("--batch", default=256, type=int)
    #parser.add_argument("--feat-wt", required=True, help="Feature Net Weights", dest="feat_wt")
    parser.add_argument("--disc-wt", required=True, help="Disc Net Weights", dest="disc_wt")
    
    parser.add_argument("--pretrained", action="store_true", help="Pretrained model")
    parser.add_argument("--net", choices=["vgg11", "squeezenet", "resnet50", "resnet18", "densenet161",
        "vgg16", "alexnet", "resnet101", "densenet121", "resnext101",
        "wideresnet101", "vgg19"],
            required=True, help="Network Architecture")
    parser.add_argument("--disc_net", choices=["linear", "resclass", "vggclass", "basenet"],
            default="linear", help="Discriminator Network Architecture")
    parser.add_argument("--split", default=0.5, type=float, help="train/test split percentage")
    parser.add_argument("--cuda", default=0, type=int, help="number of cuda device to use")
    parser.add_argument("--criterion", default="ce", choices=["bce", "ce"])
    parser.add_argument("--load-model", dest="model", default=None, type=str, help="Model file to load")
    parser.add_argument("--title", default="noname_run", help="Name for this experiment")
    parser.add_argument("--downsize", default=None, type=float, help="Downsize the primary dataset")
    parser.add_argument("--disc_classes", default=2, type=int, help="number of discriminator classes")
    parser.add_argument("--layer_probe", default="penultimate", help="what layer of feature network to probe", 
            choices=["conv1", "early_conv", "mid_conv", "late_conv", "penultimate", "logits"])
    parser.add_argument("--hidden_size", default=None, type=int, help="hidden unit size of disc network")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--samples", default=-1, type=int, help="number of samples for few show discriminator training")
    parser.add_argument("--dt", default="imagenet", help="data transform type")
    parser.add_argument("--batch-norm", dest="batch_norm", action="store_true") 
    return parser





def main(args):

    file_locs = json.load(open("datasets.json"))

    source_ds = file_locs[args.source]
    target_ds = file_locs[args.target]

    if args.source_sub is not None:
        print("Sub Directory: Source")
        source_ds = os.path.join(source_ds, args.source_sub)
        args.source = "%s_%s" %(args.source, args.source_sub)

    if args.target_sub is not None:
        print("Sub Directory: Target")
        target_ds = os.path.join(target_ds, args.target_sub)
        args.target = "%s_%s" %(args.target, args.target_sub)


    date = datetime.datetime.now()
    dataset_dir = os.path.join("experiments", args.title + "_" + date.isoformat("_"))

    os.mkdir(dataset_dir)

    arg_dict = vars(args)
    arg_file = os.path.join(dataset_dir, "args.json")
    with open(arg_file, "w+") as f:
        json.dump(arg_dict, f)

    log_file = os.path.join(dataset_dir, "log.txt")
    logging.basicConfig(level=logging.DEBUG,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            format="%(message)s")

    logging.info(str(arg_dict))
    # Cuda Device
    device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
    logging.info("Cuda Device: " + str(device))
    if args.dt == "imagenet":

        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        data_transform = None


    logging.info("Second Dataset")
    
    if args.criterion == "bce":
        criterion = nn.functional.binary_cross_entropy
    else:
        criterion = nn.functional.cross_entropy


    dataset_params = dict(
            base = [(args.source, data_transform, data_transform, args.source_sub)],
            cal = [(args.target, data_transform, data_transform, args.target_sub)]
            )

    datasets = helpers.loadDatasets(dataset_params)
    sampler_gen = helpers.samplerGenerator(
            dict(name = "intersection",
                sampler_params=dict(
                    batch_size=128,
                    random_seed=73)
                )
            )
   

    base_ds = datasets["base"]
    target_ds = datasets["cal"][0]
    base_ds = base_ds[0]

    base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], [target_ds[0]])
    target_ds_sampler = sampler_gen(target_ds[0], target_ds[1], [base_ds[0]])

    feat_net = networks._load_base_model(args.net, layer_probe=args.layer_probe, pretrained=args.pretrained)
    feat_net.eval()
    
    feat_net.to(device)
    
    disc_base = networks.DiscWrapper(args.disc_net, input_dims=feat_net.dims, output_dims=args.disc_classes)
    disc_net = networks.DiscPackage(feat_net=feat_net, disc_net=disc_base)
    
    disc_state_dict = torch.load(args.disc_wt)
    disc_state_dict = {d.replace("module.",""): k for d, k in disc_state_dict["model_state_dict"].items()}
    disc_net.load_state_dict(disc_state_dict)
    
    disc_net.eval()
    disc_net.to(device)
    
    logging.info("Evaluate Discriminator Model")

    eval_params = [("accuracy", None),
            ("auc", None),
            ("brier", None),
            ("ece", None),
            ("test_avg", None),
            ("loss", None)]

    #def disc_pass(x):
    #    x = feat_net.disc_forward(x)
    #    x = disc_net(x)
    #    return x


    de = eval_discriminator(disc_net,
            [base_ds_sampler.test, target_ds_sampler.test],
            eval_params, device=device,
            build_labels=build_disc_labels)




    logging.info(de)
    logging.info("Evaluate Discriminator Model - Finished")
    


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
