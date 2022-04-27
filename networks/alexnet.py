import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from torchvision import models

import logging

from .basenet import *




def find_alexnet_featuers(features):
    downsample = 1
    num_features = 0


    for layer in features:

        if isinstance(layer, nn.Conv2d):
            downsample = downsample * layer.stride[0] * layer.stride[1]
            num_features = layer.out_channels
        elif isinstance(layer, nn.MaxPool2d):
            downsample = downsample * layer.stride * 2

        logging.info("Downsample: %d" % downsample)
        logging.info("Features: %d" % num_features)
        return downsample, num_features



class AlexNet(FeatureWrapper):


    def __init__(self, model_type, layer_probe="penultimate", pretrained=True):
        super(AlexNet, self).__init__(model_type, layer_probe=layer_probe, pretrained=pretrained)

        valid_layer_probes = ["conv1", "early_conv", "mid_conv", "late_conv", "penultimate", "logits"]

        if layer_probe not in valid_layer_probes:
            raise ValueError("%s: not a valid layer probe" % layer_probe)

        self._build_net()

        self._set_disc_features()

    def _build_net(self):
        if self.model_type == "alexnet":
            self.net = models.alexnet(self.pretrained)
        else:
            raise ValueError("AlexNet Model Type not found: %s" % self.model_type) 


    def _set_disc_features(self):

        if self.layer_probe == "conv1":
            self.disc_features = self.net.features[0:2]

            ds, fs = find_alexnet_features(self.disc_features)
            self.dims = 220 * 220 * fs // ds 
        elif self.layer_probe == "early_conv":
            self.disc_features = self.net.features[0:6]
            ds, fs = find_alexnet_features(self.net.disc_features)
            self.dims = 208 * 208 * fs // ds
        elif self.layer_probe == "mid_conv":
            self.disc_features = self.net.features[0:8]
            ds, fs = find_alexnet_features(self.disc_features)
            self.dims = 208 * 208 * fs // ds 
        elif self.layer_probe == "late_conv":
            self.disc_features = self.net.features[0:10]
            ds, fs = find_alexnet_features(self.disc_features)
            self.dims = 208 * 208 * fs // ds 
        elif self.layer_probe == "penultimate":
            def f(x):
                out = self.net.features(x)
                out = self.net.avgpool(out)
                out = torch.flatten(out, 1)
                tmp_sequential = self.net.classifier[0:6]
                out = tmp_sequential(out)
                return out

            self.disc_features = f
            self.dims = 4096
            self.class_features = self.net.classifier[6]
        elif self.layer_probe == "logits":
            self.disc_features = self.net
            self.dims = 1000
        else:
            raise ValueError("%s: Model type at %s not defined" % (self.model_type, self.layer_probe))


        
_MODELS = {
        "alexnet": AlexNet
        }
