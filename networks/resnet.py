import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from torchvision import models

import logging

from .basenet import *




class ResNetCIFAR(nn.Module):


    def __init__(self, 
            block,
            layers,
            num_classes=10,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None):

        super(ResNetCIFAR, self).__init__()


        logging.info("Num Classes: %d" %num_classes)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                    "replace_stride_with_dilation should be None "
                    "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
                    )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )

        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )

        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x



class ResNet(FeatureWrapper):


    def __init__(self, model_type, layer_probe="penultimate", pretrained=True, num_classes=1000):
        super(ResNet, self).__init__(model_type, layer_probe=layer_probe, pretrained=pretrained)

        valid_layer_probes = ["conv1", "early_conv", "mid_conv", "late_conv", "penultimate", "logits"]

        self.num_classes = num_classes

        if layer_probe not in valid_layer_probes:
            raise ValueError("%s: not a valid layer probe" % layer_probe)

        self._build_net()

        self._set_disc_features()

    def _build_net(self):
        if self.model_type == "resnet18":
            self.net = models.resnet18(self.pretrained)
        elif self.model_type == "resnet18-cifar":
            self.net = ResNetCIFAR(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes)
        elif self.model_type == "resnet34":
            self.net = models.resnet34(self.pretrained)
        elif self.model_type == "resnet50":
            self.net = models.resnet50(self.pretrained)
        elif self.model_type == "resnet101":
            self.net = models.resnet101(self.pretrained)
        elif self.model_type == "resnet152":
            self.net = models.resnet152(self.pretrained)
        elif self.model_type == "wideresnet101":
            self.net = models.wide_resnet101_2(self.pretrained)
        elif self.model_type == "resnext101":
            self.net = models.resnext101_32x8d(self.pretrained)
        elif self.model_type == "augmix":
            self.net = models.resnet50(pretrained=True)
            
            pth = "models/augmix.pth"
            if not os.path.exists(pth):
                logging.info("downloading Augmix Model")
                import gdown
                gdown.cached_download("https://drive.google.com/u/0/uc?id=1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF", pth) #, postprocess=gdown.extractall)
                logging.info("Augmix model download completed")

            state_dict = torch.load(pth)

            new_state_dict = {d.replace("module.",""): k for d, k in state_dict["state_dict"].items()}

            self.net.load_state_dict(new_state_dict)
        elif self.model_type == "deepaugment":
            self.net = models.resnet50(pretrained=False)
            pth = "models/deepaugment.pth"
            if not os.path.exists(pth):
                import gdown
                gdown.cached_download("https://drive.google.com/u/0/uc?id=1DPRElQnBG66nd7GUphVm1t-5NroL7t7k", pth)

            state_dict = torch.load(pth)
            state_dict = torch.load(pth)

            new_state_dict = {d.replace("module.",""): k for d, k in state_dict["state_dict"].items()}

            self.net.load_state_dict(new_state_dict)
        elif self.model_type == "am-deepaugment":
            self.net = models.resnet50(pretrained=False)
            pth = "models/am-deepaugment.pth"

            if not os.path.exists(pth):
                import gdown
                gdown.cached_download("https://drive.google.com/u/0/uc?id=1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP", pth)
                
            state_dict = torch.load(pth)
            new_state_dict = {d.replace("module.",""): k for d, k in state_dict["state_dict"].items()}

            self.net.load_state_dict(new_state_dict)
        else:
            raise ValueError("ResNet Model Type not found: %s" % self.model_type) 


    def _set_disc_features(self):

        if self.layer_probe == "conv1":
            self.disc_features = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu)
            self.dims = 802816
        elif self.layer_probe == "early_conv":
            self.disc_features = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool, self.net.layer1)
            self.dims = 802816
        elif self.layer_probe == "mid_conv":
            self.disc_features = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool, 
                    self.net.layer1, self.net.layer2)
            self.dims = 401408
        elif self.layer_probe == "late_conv":
            self.disc_features = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool, 
                                               self.net.layer1, self.net.layer2, self.net.layer3)
            self.dims = 200704
        elif self.layer_probe == "penultimate":
            self.disc_features = nn.Sequential(self.net.conv1, self.net.bn1, self.net.relu, self.net.maxpool, 
                                            self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4, self.net.avgpool)
            self.dims = self.net.fc.in_features
            self.class_features = self.net.fc
        elif self.layer_probe == "logits":
            self.disc_features = self.net
            self.dims = 1000
        else:
            raise ValueError("%s: Model type at %s not defined" % (self.model_type, self.layer_probe))



        
_MODELS = {
        "resnet18": ResNet,
        "resnet18-cifar": ResNet,
        "resnet34": ResNet,
        "resnet50": ResNet,
        "resnet101": ResNet,
        "resnet152": ResNet,
        "wideresnet101": ResNet,
        "resnext101": ResNet,
        "augmix": ResNet,
        "am-deepaugment": ResNet,
        "deepaugment": ResNet
        }

