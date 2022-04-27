import torch
import networks
import logging

from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
# Figure out how to handle device
from .evaluators import *
from .helpers import *
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import torch.nn as nn



class BaseTrainer(object):
    """
    Base class for other calibration algorithms
    """

    def __init__(self):
        pass #Make Required


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None, batch_norm=False):
        self.feat_net = feat_net

    def regressor_training(self, regressor_model, cls_evals, disc_evals, train_ds,  sampler_gen):

        if "regressor_type" not in self.parameters or self.parameters["regressor_type"] == "standard":
            regressor_model, reg_results =  train_regressor(regressor_model,
                    cls_evals,
                    disc_evals,
                    self.parameters["regressor_features"],
                    self.parameters["regressor_target"],
                    self.parameters["regressor_params"],
                    self.parameters["regressor_eval_params"],
                    device=self.device)
            return regressor_model, reg_results
        elif self.parameters["regressor_type"] == "logistic":
            if isinstance(regressor_model.net, nn.Linear):
                regressor_model.net.bias = nn.Parameter(torch.tensor([1.5], device=self.device, requires_grad=True))
            regressor_model, reg_results =  train_logistic_regressor(regressor_model,
                    cls_evals,
                    disc_evals,
                    self.parameters["regressor_features"],
                    self.parameters["regressor_target"],
                    self.parameters["regressor_params"],
                    self.parameters["regressor_eval_params"],
                    device=self.device)
            return regressor_model, reg_results
        elif self.parameters["regressor_type"] == "dynamic_temperature":
            if isinstance(regressor_model.net, nn.Linear):
                regressor_model.net.bias = nn.Parameter(torch.tensor([1.5], device=self.device, requires_grad=True))

            regressor_model, reg_results = train_dynamic_temperature(regressor_model,
                   self.feat_net,
                   train_ds, 
                   cls_evals,
                   disc_evals, 
                   self.parameters["regressor_features"],
                   self.parameters["regressor_target"],
                   self.parameters["regressor_params"],
                   self.parameters["regressor_eval_params"],
                   sampler_gen,
                   device=self.device)

            return regressor_model, reg_results
        elif self.parameters["regressor_type"] == "acc_diff":
            if self.parameters["regressor_features"][0] != "test_avg" and self.parameters["regressor_features"][0] != "base_acc":
                raise ValueError("Only Average Max Probs allowed at this point")

            regressor_model, reg_results = fixed_accuracy_feat_diff(regressor_model,
                    cls_evals,
                    disc_evals,
                    self.parameters["regressor_features"],
                    self.parameters["regressor_target"],
                    self.parameters["regressor_params"],
                    self.parameters["regressor_eval_params"],
                    sampler_gen,
                    device=self.device)

            return regressor_model, reg_results
        elif self.parameters["regressor_type"] == "acc_raw":
            if self.parameters["regressor_features"][0] != "test_avg" and  self.parameters["regressor_features"][0] != "base_acc":
                raise ValueError("Only Average Max Probs allowed at this point")

            regressor_model, reg_results = fixed_accuracy_raw(regressor_model,
                    cls_evals,
                    disc_evals,
                    self.parameters["regressor_features"],
                    self.parameters["regressor_target"],
                    self.parameters["regressor_params"],
                    self.parameters["regressor_eval_params"],
                    sampler_gen,
                    device=self.device)

            return regressor_model, reg_results
        elif self.parameters["regressor_type"] == "dist_diff":
            pass
        else:
            raise ValueError("Unknown Regressor Type: %s" %(self.parameters["regressor_type"]))


    def group_regressor_eval(self, cls_evals, disc_evals, base_ds, 
            test_ds, sampler_gen):
        """
        Group Based Prediction/Evaluation of model performance
        """

        regressor_features = self.parameters["regressor_features"]
        regressor_target = self.parameters["regressor_target"]

        features = build_single_feature(cls_evals, disc_evals,
                       regressor_features)

        features = features.to(self.device)
        
        if "regressor_type" not in self.parameters or self.parameters["regressor_type"] == "standard":
            with torch.no_grad():
                pred = self.regressor_model(features)
                pred = pred.cpu().item()
        elif self.parameters["regressor_type"] == "logistic":
            with torch.no_grad():
                pred = self.regressor_model(features)
                pred = pred.cpu().item()

            pred = cls_evals[0][regressor_target] - pred
        elif self.parameters["regressor_type"] == "dynamic_temperature":
            pred_accuracy = AvgMaxScore(raw_scores=True)
            bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0],
                             test_ds[0])
            
            eval_params = [("test_avg", None)]

            test_ds_sampler = sampler_gen(test_ds[0], test_ds[1], [base_ds[0]])

            with torch.no_grad():
                temperature = self.regressor_model(features)
                eval_results = eval_classifier(self.feat_net,
                        test_ds_sampler.all,
                        eval_params,
                        temperature=temperature,
                        device=self.device,
                        filtering=targetf,
                        conversion=tc)

                pred_avg = eval_results["test_avg"]
                pred = cls_evals[0][regressor_target] - pred_avg
        elif self.parameters["regressor_type"] == "acc_diff":

            for reg_feat in regressor_features:
                pred = cls_evals[0][reg_feat] - cls_evals[1][reg_feat]

        elif self.parameters["regressor_type"] == "acc_raw":
            for reg_feat in regressor_features:
                if reg_feat == "base_acc":
                    pred = cls_evals[0]["accuracy"] - cls_evals[0]["accuracy"]
                else:
                    pred = cls_evals[0]["accuracy"] - cls_evals[1][reg_feat]

        else:
            raise ValueError("Regressor Type: %s not found" % self.parameters["regressor_type"])

        return pred




    def instance_prediction(self, x, cls_eval=None):
        """
        Instance Prediction scores shifted based on calibrations
        """
        pass
    
    def group_prediction(self, test_ds, sampler_gen):
        """
        Group Prediction for a distributional split
        """
        pass


