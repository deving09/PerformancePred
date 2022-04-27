import torch
import networks
import logging

import numpy as np

from torch.optim.lr_scheduler import StepLR

import torch.optim as optim

from .evaluators import *
from torch.utils.data import DataLoader, TensorDataset
#from .discriminator import build_optimizer
#from .discriminator import train_regressor
from .helpers import *
from .basetrainer import *

import utils as d_utils


class VanillaTrain(BaseTrainer):
    """
    Vanilla Training for BaseTrainer
    """

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming
        self.conn = conn


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None, batch_norm=False):
        base_ds = base_ds[0]

        # Initialize feat net
        self.feat_net = feat_net
        self.feat_net.to(self.device)

        logging.info(self.feat_net.dims)
        self.base_ds = base_ds

        logging.info("Training Vanilla Model")

        logging.info(base_ds[0])
        
        base_ds_sampler = sampler_gen(base_ds[0], base_ds[0], [])
        #logging.info(base_ds_sampler)

        epochs = self.parameters["train_params"]["epochs"]
        criterion = self.parameters["train_params"]["criterion"]
        scheduler_params = self.parameters["train_params"]["scheduler"]
        
        optimizer = build_optimizer(feat_net,
                disc_net=None,
                train_params=self.parameters["train_params"]["optimizer"])

        scheduler = StepLR(optimizer,
            step_size=scheduler_params["step_size"],
            gamma=scheduler_params["gamma"])

        cnt = 0
        running_loss = 0.0
        best_acc = 0.0
        valid_loss = 0.0
        acc = 0.0

        ds_train = base_ds_sampler.train
        logging.info(f"len train data: {len(ds_train)}")
        ds_val = base_ds_sampler.val
        logging.info(f"len test data: {len(ds_val)}")

        for epoch in tqdm(range(epochs)):

            if batch_norm:
                feat_net.train()
            else:
                feat_net.eval()

            for i, (inputs, labels) in enumerate(ds_train):

                inputs  = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = feat_net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if cnt % 20 == 19:
                    logging.info("[%d, %5d] loss: %.3f" %(epoch + 1,
                        cnt +1 , running_loss / 20))
                    running_loss = 0.0

            valid_loss = 0.0
            acc = 0.0

            # Validation Step
            logging.info("Epoch: %d" % epoch)
            acc_calc = AccuracyCalculator()
            loss_calc = LossAggregator(criterion)

            feat_net.eval()
            with torch.no_grad():
                for i, (test_inputs, targets) in enumerate(ds_val):
                   
                    test_inputs = test_inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = feat_net(test_inputs)

                    acc_calc.update(outputs, targets)
                    loss_calc.update(outputs, targets)


            acc = acc_calc.results()
            a_loss = loss_calc.results()

            logging.info("Test vanilla Acc: %f\t Loss: %f" % (acc, a_loss))
            
            scheduler.step()

        logging.info("Model Finished Training")

        return feat_net, {}



