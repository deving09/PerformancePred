import torch
import networks
import logging


from .evaluators import *
from .helpers import *
from torch.utils.data import DataLoader, TensorDataset


class BootstrapAlg(BaseTrainer):
    """
    Saves values by instance name for repeat inference
    """

    def __init__(self, parameters, device=None, model_naming=None):

        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token):

        base_ds = base_ds[0]
