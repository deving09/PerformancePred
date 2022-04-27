import torch

from .data import *
from .db import *

import sqlite3
from sqlite3 import Error

from .standarddataset import *
from .imageneta import *
from .imagenetr import *
from .imagenetc import *
from .imagenetv2 import *
from .imagenetvidrobust import *
from .cifar import *



def loadDatasets(datasets_config):

    train_ds = []

    for i, train_cls_name in datasets_config["train"]:
        train_ds.append(vars(train_cls_name)(*dataset_config["loader_params"]))

    test_ds = []

    for i, test_cls_name in datasets_config["test"]:
        test_ds.append(vars(test_cls_name)(test=True))


    return train_ds, test_ds

def modelLoader(datasets_model):

    return vars(datasets_model["model"])(*dataset_config["loader_params"])


def trainAlgorithm(base_algorithm):
    
    return vars(base_algorithm["algorithm"])(*base_algorithm["algorithm_params"])

def calibrationAlgorithm(cal_algorithm):

    return vars(cal_algorithm["algorithm"])(*cal_algorithm["algorithm_params"])



def accuracy_1(net, data_loader, device, criterion, class_net=False):
    correct = 0.0
    total = 0.0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if class_net:
                outputs = net(images)
            else:
                outputs = net.disc_forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            outputs.detach()
            lt = criterion(outputs, labels)
            loss_total = loss_total + lt

    return correct/total, loss_total/total
