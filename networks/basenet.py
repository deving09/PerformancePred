import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from torchvision import models

import logging

class BaseNet(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size=32):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



class VGGClassifier(nn.Module):

    def __init__(self, input_dims=(512*7*7), num_classes=2, hidden_size=4096):
        super(VGGClassifier, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden_size, num_classes)
                )

    def forward(self, x):
        return self.net(x)

class ResNetClassifier(nn.Module):

    def __init__(self, input_dims, num_classes=2, hidden_size=512):
        super(ResNetClassifier, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, num_classes),
                )

    def forward(self, x):
        return self.net(x)


class EnsembleModel(nn.Module):

    def __init__(self, models, aggregate=None):
        super(EnsembleModel, self).__init__()
        
        self.ens_len = len(models)

        for i,m in  enumerate(models):
            tmp_str = "model_%d" % i
            self.tmp_str = m

    
    def forward(self, x):
        res_list = []
        for i in range(self.ens_len):
            tmp_str = "model_%d" %i
            res_list.append(self.tmp_str(x))
        return torch.stack(res_list, 2)

    def combine_preds(self, preds, temperature=1.0, conv=None):

        probs = torch.nn.functional.softmax(preds, dim=1)
        final_preds = torch.mean(probs, 2)
        new_logits =  torch.log(final_preds) + 1.0

        return new_logits


    def combine_preds_eval(self, preds, targets, conv=None):

        new_preds = []
        preds = preds.cpu()
        if conv is not None:

            for i in range(preds.shape[-1]):
                p = preds[:, :, i]
                p, trg = conv(p, targets)
                new_preds.append(p)

            targets = trg
            preds = torch.stack(new_preds, 2)
        
        probs = torch.nn.functional.softmax(preds, dim=1)
        final_preds = torch.mean(probs, 2)
        new_logits =  torch.log(final_preds) + 1.0

        return new_logits, targets

class DiscPackage(nn.Module):

    def __init__(self, feat_net=None, disc_net=None):
        super(DiscPackage, self).__init__()

        self.feat_net = feat_net
        self.disc_net = disc_net

    def forward(self, x):
        x = self.feat_net.disc_forward(x)
        x = self.disc_net(x)
        return x


class DiscWrapper(nn.Module):

    def __init__(self, model_type, input_dims, output_dims, hidden_size=None, flatten=None):
        super(DiscWrapper, self).__init__()


        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model_type = model_type

        valid_model_types = ["basenet", "resclass", "vggclass", "linear"]
        
        if model_type not in valid_model_types:
            raise ValueError("%s: not a valid model type" % model_type)

        self.hidden_size = hidden_size
        self.flatten = flatten
        self._load_model()


    def _load_model(self):

        if self.model_type == "basenet":
            if self.hidden_size is not None:
                self.net = BaseNet(self.input_dims, self.output_dims, hidden_size=self.hidden_size)
            else:
                self.net = BaseNet(self.input_dims, self.output_dims)
        elif self.model_type == "resclass":
            if self.hidden_size is not None:
                self.net = ResNetClassifier(self.input_dims, self.output_dims, hidden_size=self.hidden_size)
            else:
                self.net = ResNetClassifier(self.input_dims, self.output_dims)
        elif self.model_type == "vggclass":
            if self.hidden_size is not None:
                self.net = VGGClassifier(self.input_dims, self.output_dims, hidden_size=self.hidden_size)
            else:
                self.net = VGGClassifier(self.input_dims, self.output_dims)
        elif self.model_type == "linear":
            self.net = nn.Linear(self.input_dims, self.output_dims)
        else:
            raise ValueError("Model Type: %s not supported" % self.model_type)


        if self.flatten is None:
            if self.model_type in ["basenet", "resclass", "vggclass", "linear"]:
                self.flatten = True
            elif self.model_type in ["convbase"]:
                self.flatten = False


    def forward(self, x):
        if self.flatten:
            x = torch.flatten(x, 1)

        return self.net(x)





class IdentityNet(nn.Module):

    def __init__(self, dims=1000):
        self.dims = dims
        super(IdentityNet, self).__init__()
        self.net = None

    def forward(self, x):
        return x

    def disc_forward(self, x):
        return x


class BootstrapNet(nn.Module):

    def __init__(self, net, flatten=True, dims=None):
        super(BootstrapNet, self).__init__()
        self.net = net
        self.dims = dims

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)

    def disc_forward(self, x):
        return x
        


class FeatureWrapper(nn.Module):

    def __init__(self, model_type,  layer_probe="penultimate", pretrained=True):
        super(FeatureWrapper, self).__init__()

        self.model_type = model_type
        self.layer_probe = layer_probe
        self.pretrained = pretrained


    def forward(self, x):
        return self.net(x)

    def disc_forward(self, x):
        return self.disc_features(x)

    def class_forward(self, x):
        return self.class_features(x)



_MODELS = {"basenet":BaseNet}

