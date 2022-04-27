import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from torchvision import models

import logging

from .basenet import *




def find_densenet_features(features):
    import torchvision.models.densenet as densenet
    downsample = 1
    num_features = 0


    for layer in features:
        if isinstance(layer, nn.Conv2d):
            downsample  = downsample * layer.stride[0] * layer.stride[1]
            num_features += layer.out_channels
        elif isinstance(layer, densenet._DenseBlock):
            for l in layer.values():
                num_features += l.conv2.out_channels
        elif isinstance(layer, nn.MaxPool2d):
            downsample = downsample * layer.stride * 2
        elif isinstance(layer, densenet._Transition):
            num_features = layer.conv.out_channels
            downsample = downsample * 2 * layer.pool.stride

    logging.info("Downsample: %d" % downsample)
    logging.info("Features: %d" % num_features)
    return downsample, num_features



class DenseNet(FeatureWrapper):


    def __init__(self, model_type, layer_probe="penultimate", pretrained=True):
        super(DenseNet, self).__init__(model_type, layer_probe=layer_probe, pretrained=pretrained)

        valid_layer_probes = ["conv1", "early_conv", "mid_conv", "late_conv", "penultimate", "logits"]

        if layer_probe not in valid_layer_probes:
            raise ValueError("%s: not a valid layer probe" % layer_probe)

        self._build_net()

        self._set_disc_features()

    def _build_net(self):
        if self.model_type == "densenet121":
            self.net = models.densenet121(self.pretrained)
        elif self.model_type == "densenet161":
            self.net = models.densenet161(self.pretrained)
        else:
            raise ValueError("ResNet Model Type not found: %s" % self.model_type) 


    def _set_disc_features(self):

        if self.layer_probe == "conv1":
            self.disc_features = self.net.features[0:3]
            ds, fs =  find_densenet_features(self.disc_features)
            self.dims = 224 * 224 * fs // ds
        elif self.layer_probe == "early_conv":
            self.disc_features = self.net.features[0:5]
            ds, fs =  find_densenet_features(self.disc_features)
            self.dims = 224 * 224 * fs // ds
        elif self.layer_probe == "mid_conv":
            self.disc_features = self.net.features[0:7]
            ds, fs =  find_densenet_features(self.disc_features)
            self.dims = 224 * 224 * fs // ds
        elif self.layer_probe == "late_conv":
            self.disc_features = self.net.features[0:9]
            ds, fs =  find_densenet_features(self.disc_features)
            self.dims = 224 * 224 * fs // ds
        elif self.layer_probe == "penultimate":
            def f(x):
                out = self.net.features(x)
                out = F.relu(out, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1,1))
                return out
            self.disc_features = f
            self.dims = self.net.classifier.in_features
            self.class_features = self.net.classifier
        elif self.layer_probe == "logits":
            self.disc_features = self.net
            self.dims = 1000
        else:
            raise ValueError("%s: Model type at %s not defined" % (self.model_type, self.layer_probe))


 
_MODELS = {
        "densenet161": DenseNet,
        "densenet121": DenseNet
        }
