import torch
import numpy as np

import logging

import networks
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
# Figure out how to handle device
from .evaluators import *
from .helpers import *
from .basetrainer import *

import utils as d_utils

from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

def calc_mean(sampler, feat_net, device=None):

    cnt = 0.0

    with torch.no_grad():
        for i, samples in tqdm(enumerate(sampler.all)):

            inputs, labels = samples

            inputs = inputs.to(device)

            batch_size = inputs.size(0)

            if i == 0:
                feats = feat_net.disc_forward(inputs)
                logging.info(feats.shape)
                feat_mean = feats.mean(0)
                logging.info(feat_mean.shape)
            else:
                curr_mean = feat_net.disc_forward(inputs).mean(0)
                feat_mean = feat_mean * (cnt/(cnt + batch_size)) + curr_mean * (batch_size/(cnt + batch_size))

            cnt += batch_size

    logging.info("Feat Mean: " + str(feat_mean.shape))

    return feat_mean


def calc_cov(sampler, feat_net, f_mean, device=None):

    cnt = 0.0
    with torch.no_grad():
        for i, samples in tqdm(enumerate(sampler.all)):

            inputs, labels = samples
            inputs = inputs.to(device)

            batch_size = inputs.size(0)

            if i == 0:
                feats = feat_net.disc_forward(inputs)
                logging.info(feats.shape)
                feats_diff = feats - f_mean
                feats_diff_t = torch.transpose(feats_diff, 1, -1)
                logging.info(feats_diff_t.shape)
                logging.info(feats_diff.shape)
                feats_cov = torch.matmul(feats_diff, feats_diff_t).mean(0)
                logging.info(feats_cov.shape)
            else:
                feats = feat_net.disc_forward(inputs)
                feats_diff = feats - f_mean
                feats_diff_t = torch.transpose(feats_diff, 1, -1)
                curr_cov = torch.matmul(feats_diff, feats_diff_t).mean(0)
                feats_cov = feats_cov * (cnt/(cnt + batch_size)) + curr_cov *(batch_size/(cnt + batch_size))

            cnt += batch_size

    return feats_cov.squeeze()



def identity_mmd(sampler_a, sampler_b, feat_net, device=None):
    a_mean = calc_mean(sampler_a, feat_net, device=device)
    b_mean = calc_mean(sampler_b, feat_net, device=device)
    
    mean_dist_sq = torch.pow(a_mean - b_mean, 2).sum(0)
    return mean_dist_sq


def identity_mmd_batch(sampler_a, sampler_b, feat_net, device=None):
    ds_group = [sampler_a.all, sampler_b.all]

    mean_dist_sq = 0.0
    with torch.no_grad():
        for i, samples in tqdm(enumerate(zip(*ds_group))):

            samples_a, l_a = samples[0]
            samples_b, l_b = samples[1]

            a_mean = feat_net.disc_forward(samples_a).mean(0)
            b_mean = feat_net.disc_forward(samples_b).mean(0)

            curr_dist = torch.pow(a_mean - b_mean, 2).sum(0)
            mean_dist_sq = i * mean_dist_sq / (i+1) + curr_dist / (i+1)

    return mean_dist_sq

def l2_kernel_mmd(sampler_a, sampler_b, feat_net, device=None):
    ds_group = [sampler_a.all, sampler_b.all]

    mean_dist_sq = 0.0
    with torch.no_grad():
        for i, samples in tqdm(enumerate(zip(*ds_group))):

            samples_a, l_a = samples[0]
            samples_b, l_b = samples[1]

            a_feats = feat_net.disc_forward(samples_a)
            b_feats = feat_net.disc_forward(samples_b)


            a_len = a_feats.shape[0]
            b_len = b_feats.shape[0]

            h_val = 0.0


            if a_len != b_len:
                continue
            
            top_map = [i // b_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            bottom_map = [i % a_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            
            top_map = torch.tensor(top_map, dtype=torch.long, device=device)
            bottom_map = torch.tensor(bottom_map, dtype=torch.long, device=device)
            #to
            a_top = torch.index_select(a_feats, 0, top_map)
            a_bottom = torch.index_select(a_feats, 0, bottom_map)
            
            b_top = torch.index_select(b_feats, 0, top_map)
            b_bottom = torch.index_select(b_feats, 0, bottom_map)
            #b_samp = torch.index_select(b_feats, 0, torch.tensor(bottom_map, dtype=torch.LongTensor))

            h_val += torch.pow(a_top - a_bottom, 2).sum()
            h_val += torch.pow(b_top - b_bottom, 2).sum()

            h_val -= torch.pow(a_top - b_bottom, 2).sum()
            h_val -= torch.pow(a_bottom - b_top, 2).sum()

            h_val = h_val / (a_len *(b_len -1))
            
            #curr_dist = torch.pow(a_mean - b_mean, 2).sum(0)
            mean_dist_sq = i * mean_dist_sq / (i+1) + h_val / (i+1)

    return mean_dist_sq


def dot_kernel_mmd(sampler_a, sampler_b, feat_net, device=None):
    ds_group = [sampler_a.all, sampler_b.all]

    mean_dist_sq = 0.0

    with torch.no_grad():

        def k(x, y):
            d = (x * y).sum()
            return d

        for i, samples in tqdm(enumerate(zip(*ds_group))):

            samples_a, l_a = samples[0]
            samples_b, l_b = samples[1]

            a_feats = feat_net.disc_forward(samples_a)
            b_feats = feat_net.disc_forward(samples_b)


            a_len = a_feats.shape[0]
            b_len = b_feats.shape[0]

            h_val = 0.0


            if a_len != b_len:
                continue
            
            top_map = [i // b_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            bottom_map = [i % a_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            
            top_map = torch.tensor(top_map, dtype=torch.long, device=device)
            bottom_map = torch.tensor(bottom_map, dtype=torch.long, device=device)
            
            a_top = torch.index_select(a_feats, 0, top_map)
            a_bottom = torch.index_select(a_feats, 0, bottom_map)
            
            b_top = torch.index_select(b_feats, 0, top_map)
            b_bottom = torch.index_select(b_feats, 0, bottom_map)

            h_val += k(a_top, a_bottom)
            h_val += k(b_top, b_bottom)
            h_val -= k(a_top, b_bottom)
            h_val -= k(b_top, a_bottom)
            
            h_val = h_val / (a_len *(b_len -1))
            
            mean_dist_sq = i * mean_dist_sq / (i+1) + h_val / (i+1)

    return mean_dist_sq



def gaussian_kernel_mmd(sampler_a, sampler_b, feat_net, device=None, sigma=5.0):
    ds_group = [sampler_a.all, sampler_b.all]

    mean_dist_sq = 0.0

    with torch.no_grad():

        def k(x, y):
            gap = torch.sqrt(torch.pow(x - y, 2).sum(1))
            d = torch.exp(-gap / sigma).sum()
            return d

        for i, samples in tqdm(enumerate(zip(*ds_group))):

            samples_a, l_a = samples[0]
            samples_b, l_b = samples[1]

            a_feats = feat_net.disc_forward(samples_a)
            b_feats = feat_net.disc_forward(samples_b)


            a_len = a_feats.shape[0]
            b_len = b_feats.shape[0]

            h_val = 0.0


            if a_len != b_len:
                continue
            
            top_map = [i // b_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            bottom_map = [i % a_len for i in range(a_len * b_len) if i // b_len != i % a_len]
            
            top_map = torch.tensor(top_map, dtype=torch.long, device=device)
            bottom_map = torch.tensor(bottom_map, dtype=torch.long, device=device)
            
            a_top = torch.index_select(a_feats, 0, top_map)
            a_bottom = torch.index_select(a_feats, 0, bottom_map)
            
            b_top = torch.index_select(b_feats, 0, top_map)
            b_bottom = torch.index_select(b_feats, 0, bottom_map)

            h_val += k(a_top, a_bottom)
            h_val += k(b_top, b_bottom)
            h_val -= k(a_top, b_bottom)
            h_val -= k(b_top, a_bottom)
            
            h_val = h_val / (a_len *(b_len -1))
            
            mean_dist_sq = i * mean_dist_sq / (i+1) + h_val / (i+1)

    return mean_dist_sq




class MaximumMeanDiscrepancy(BaseTrainer):
    """
    MMD calculator
    """

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming
        self.sampler_gen = None
        self.conn = conn

        if "mmd_type" in self.parameters:
            self.mmd_type = self.parameters["mmd_type"]
        else:
            self.mmd_type = "identity"

        logging.info("MMD Type: %s" % self.mmd_type)


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None, batch_norm=False):

        base_ds = base_ds[0]

        # Do this in initialization
        self.feat_net = feat_net
        self.feat_net.to(self.device)

        self.base_ds = base_ds
        self.sampler_gen = sampler_gen

        base_ds_sampler = sampler_gen(base_ds[0], base_ds[0], [])

        disc_evals = []
        cls_evals  = []
        
        with torch.no_grad():
            for target_ds in train_ds:

                sigma = self.parameters.get("sigma", 5.0)

                if "gaussian" in self.mmd_type:
                    dist_full = "mmd_" + self.mmd_type + "_sigma_%.3f" % sigma
                    
                    disc_eval_token = token_maker(base_token, "mmd",
                                          self.mmd_type,
                                          sigma,
                                          base_ds, target_ds,
                                          sampler_gen)
                else:
                
                    dist_full = "mmd_" + self.mmd_type 
                    disc_eval_token = token_maker(base_token, "mmd",
                                          self.mmd_type,
                                          base_ds, target_ds,
                                          sampler_gen)

                # PARAMETER NAMINGS
                base_ds_name = self.base_ds[0].name
                if self.base_ds[0].sub_name is not None:
                    base_ds_name += "_" + self.base_ds[0].sub_name

                target_ds_name = target_ds[0].name
                if target_ds[0].sub_name is not None:
                    target_ds_name += "_" + target_ds[0].sub_name

                #dist_full = "mmd_" + self.mmd_type


                if d_utils.db.check_existence("distances", {"id":disc_eval_token}, conn=self.conn):
                    logging.info("FOUND IT")
                    disc_eval = d_utils.db.get_selection_table("distances", 
                            match_e={"id":disc_eval_token}, conn=self.conn)
                    disc_eval = disc_eval[0]
                    disc_eval["target_ds"] = target_ds_name
                    disc_eval["base_ds"] = base_ds_name
                    disc_eval["model"] = self.model_naming
                    disc_eval["dist_type"] = dist_full
                    
                    logging.info("MMD Distance: %f" % disc_eval["score"])
                    check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
                else:
                    logging.info("DIDN'T FIND IT")
                    target_ds_sampler = sampler_gen(target_ds[0], target_ds[0], [])

                    if self.mmd_type == "identity":
                        mmd_dist = identity_mmd(base_ds_sampler, target_ds_sampler, feat_net, self.device)
                        mmd_dist = mmd_dist.cpu().item()
                    elif self.mmd_type == "identity_batch":
                        mmd_dist = identity_mmd_batch(base_ds_sampler, target_ds_sampler, feat_net, self.device)
                        mmd_dist = mmd_dist.cpu().item()
                    elif self.mmd_type == "gaussian_kernel":
                        sigma = self.parameters.get("sigma", 5.0)
                        mmd_dist = gaussian_kernel_mmd(base_ds_sampler, target_ds_sampler, feat_net, self.device, sigma=sigma)
                        mmd_dist = mmd_dist.cpu().item()
                    elif self.mmd_type == "l2_kernel":
                        mmd_dist = l2_kernel_mmd(base_ds_sampler, target_ds_sampler, feat_net, self.device)
                        mmd_dist = mmd_dist.cpu().item()
                    elif self.mmd_type == "dot_kernel":
                        mmd_dist = dot_kernel_mmd(base_ds_sampler, target_ds_sampler, feat_net, self.device)
                        mmd_dist = mmd_dist.cpu().item()
                    else:
                        raise ValueError("mmd type not specified")

                    logging.info("MMD Distance: %f" % mmd_dist)

                    disc_eval = {"score": mmd_dist}
                    disc_eval["id"] =  disc_eval_token
                    disc_eval["target_ds"] = target_ds_name
                    disc_eval["base_ds"] = base_ds_name
                    disc_eval["model"] = self.model_naming
                    disc_eval["dist_type"] = dist_full
                    check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)

                
                disc_evals.append(disc_eval)

                bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0], target_ds[0])

                if bf is not None or bc is not None:
                    cls_eval_token_0 = token_maker(base_token, base_ds, sampler_gen,
                                          self.parameters["cls_eval_params"],
                                          target_ds)
                else:
                    cls_eval_token_0 = token_maker(base_token, base_ds, sampler_gen,
                                          self.parameters["cls_eval_params"])

                if d_utils.db.check_existence("classifications", {"id":cls_eval_token_0}, conn=self.conn):

                    logging.info("Found Base Classifier")
                    logging.info("Token: %s" %str(cls_eval_token_0))
                    ce_0 = d_utils.db.get_selection_table("classifications",
                            match_e={"id":cls_eval_token_0},
                            conn=self.conn)
                    ce_0 = ce_0[0]
                    ce_0["base_ds"] = target_ds_name
                    ce_0["target_ds"] = base_ds_name
                    ce_0["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
                else:
                    logging.info("Computing Base Classifier")
                    ce_0 = eval_classifier(self.feat_net, base_ds_sampler.all,
                              self.parameters["cls_eval_params"],
                              device=self.device,
                              filtering=bf,
                              conversion=bc)
                    
                    ce_0["id"] = cls_eval_token_0
                    ce_0["base_ds"] = target_ds_name
                    ce_0["target_ds"] = base_ds_name
                    ce_0["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
                
                if targetf is not None or tc is not None:
                    cls_eval_token_1 = token_maker(base_token, target_ds, sampler_gen,
                                       self.parameters["cls_eval_params"],
                                       base_ds)
                else:
                    cls_eval_token_1 = token_maker(base_token, target_ds, sampler_gen,
                                       self.parameters["cls_eval_params"])

                if d_utils.db.check_existence("classifications", {"id":cls_eval_token_1}, conn=self.conn):
                    logging.info("Found Target Classifier")
                    ce_1 = d_utils.db.get_selection_table("classifications",
                            match_e={"id":cls_eval_token_1},
                            conn=self.conn)
                    ce_1 = ce_1[0]
                    ce_1["base_ds"] = base_ds_name
                    ce_1["target_ds"] = target_ds_name
                    ce_1["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)
                    logging.info(ce_1)
                else:
                    logging.info("Computing Target Classifier")
                    ce_1 = eval_classifier(self.feat_net, target_ds_sampler.all,
                            self.parameters["cls_eval_params"], 
                            device=self.device,
                            filtering=targetf,
                            conversion=tc)

                    ce_1["id"] = cls_eval_token_1
                    ce_1["base_ds"] = base_ds_name
                    ce_1["target_ds"] = target_ds_name
                    ce_1["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)

                cls_evals.append([ce_0, ce_1])


        logging.info(disc_evals)
        logging.info(cls_evals)

        logging.info("Class and Disc Evals above")

        regressor_model = networks.DiscWrapper(self.parameters["regressor_model"]["model_type"],
                input_dims=len(self.parameters["regressor_features"]),
                output_dims=self.parameters["regressor_model"]["output_dims"],
                flatten=False)

        regressor_model.to(self.device)
        
        regressor_model, reg_results = self.regressor_training(regressor_model, cls_evals, disc_evals,
                train_ds, sampler_gen)
         

        self.regressor_model = regressor_model

        ## Do Eval Generic stuff
        ##

        base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], []) #, *self.parameters["sampler_params"])
        
        cls_eval_token = token_maker(base_token, base_ds, sampler_gen,
                             self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            cls_eval = d_utils.db.get_selection_table("classifications",
                    match_e={"id": cls_eval_token},
                    conn=self.conn)
            cls_eval = cls_eval[0]
            self.cls_eval = cls_eval
        else:

            self.cls_eval = eval_classifier(self.feat_net, 
                    base_ds_sampler.all, self.parameters["cls_eval_params"],
                    device=self.device)

            self.cls_eval["id"] = cls_eval_token
            check = d_utils.db.insert_into_table("classifications", self.cls_eval, conn=self.conn)

            

        return reg_results


    def group_prediction(self, test_ds, sampler_gen, base_token=None, cal_token=None):
        """
        Group prediction with MMD distance
        """

        base_ds_sampler = sampler_gen(self.base_ds[0], self.base_ds[1], [test_ds[0]])#, *self.parameters["sampler_params"])
        test_ds_sampler = sampler_gen(test_ds[0], test_ds[1], [self.base_ds[0]]) #, *self.parameters["sampler_params"])


        # PARAMETER NAMINGS
        base_ds_name = self.base_ds[0].name
        if self.base_ds[0].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name


        #Update to this section complete
        sigma = self.parameters.get("sigma", 5.0)

        if "gaussian" in self.mmd_type:
            dist_full = "mmd_" + self.mmd_type + "_sigma_%.3f" % sigma
            
            disc_eval_token = token_maker(base_token, "mmd",
                                  self.mmd_type,
                                  sigma,
                                  self.base_ds, test_ds,
                                  sampler_gen)
        else:
            dist_full = "mmd_" + self.mmd_type 
            disc_eval_token = token_maker(base_token, "mmd",
                                  self.mmd_type,
                                  self.base_ds, test_ds,
                                  sampler_gen)

        if d_utils.db.check_existence("distances", {"id": disc_eval_token}, conn=self.conn):
            logging.info("distance found: %s" %disc_eval_token)
            disc_eval = d_utils.db.get_selection_table("distances",
                    match_e={"id": disc_eval_token},
                    conn=self.conn)
            disc_eval = disc_eval[0]
            disc_eval["base_ds"] = base_ds_name
            disc_eval["target_ds"] = target_ds_name
            disc_eval["model"] = self.model_naming
            disc_eval["dist_type"] = dist_full
            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
        else:
            
            if self.mmd_type == "identity":
                mmd_dist = identity_mmd(base_ds_sampler, test_ds_sampler, self.feat_net, self.device)
                mmd_dist = mmd_dist.cpu().item()
            elif self.mmd_type == "identity_batch":
                mmd_dist = identity_mmd_batch(base_ds_sampler, test_ds_sampler, self.feat_net, self.device)
                mmd_dist = mmd_dist.cpu().item()
            elif self.mmd_type == "gaussian_kernel":
                sigma = self.parameters.get("sigma", 5.0)
                mmd_dist = gaussian_kernel_mmd(base_ds_sampler, test_ds_sampler, self.feat_net, self.device, sigma=sigma)
                mmd_dist = mmd_dist.cpu().item()
            elif self.mmd_type == "l2_kernel":
                mmd_dist = l2_kernel_mmd(base_ds_sampler, test_ds_sampler, self.feat_net, self.device)
                mmd_dist = mmd_dist.cpu().item()
            elif self.mmd_type == "dot_kernel":
                mmd_dist = dot_kernel_mmd(base_ds_sampler, test_ds_sampler, self.feat_net, self.device)
                mmd_dist = mmd_dist.cpu().item()


            
            logging.info("MMD Distance: %f" %mmd_dist)

            disc_eval = {"score": mmd_dist}
            disc_eval["id"] = disc_eval_token
            disc_eval["base_ds"] = base_ds_name
            disc_eval["target_ds"] = target_ds_name
            disc_eval["model"] = self.model_naming
            disc_eval["dist_type"] = dist_full
            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)


        
        bf, bc, targetf, tc = get_filters_and_conversions(self.base_ds[0], test_ds[0])

        if bf is not None or bc is not None:
            cls_eval_token = token_maker(base_token, self.base_ds, sampler_gen,
                                 self.parameters["cls_eval_params"],
                                 test_ds)
            pass
        else:
            cls_eval_token = token_maker(base_token, self.base_ds, sampler_gen,
                                 self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            logging.info("Found Base Classifier")
            cls_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":cls_eval_token},
                        conn=self.conn)
            cls_eval = cls_eval[0]
            cls_eval["base_ds"] = target_ds_name
            cls_eval["target_ds"] = base_ds_name
            cls_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)
            logging.info(cls_eval)
        else:
            logging.info(bf)
            logging.info(bc)
            cls_eval = eval_classifier(self.feat_net,
                    base_ds_sampler.all,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=bf,
                    conversion=bc)

            cls_eval["id"] = cls_eval_token
            cls_eval["base_ds"] = target_ds_name
            cls_eval["target_ds"] = base_ds_name
            cls_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)

        if targetf is not None or tc is not None:
            test_eval_token = token_maker(base_token, test_ds, sampler_gen,
                                   self.parameters["cls_eval_params"],
                                   self.base_ds)
        else:
            test_eval_token = token_maker(base_token, test_ds, sampler_gen,
                                   self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":test_eval_token}, conn=self.conn):
            logging.info("group prediction test token: " + test_eval_token)
            test_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":test_eval_token}, 
                        conn=self.conn)
            test_eval = test_eval[0]
            test_eval["base_ds"] = base_ds_name
            test_eval["target_ds"] = target_ds_name
            test_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)
        else:
            test_eval = eval_classifier(self.feat_net,
                    test_ds_sampler.test,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=targetf,
                    conversion=tc)
            
            test_eval["id"] = test_eval_token
            test_eval["base_ds"] = base_ds_name
            test_eval["target_ds"] = target_ds_name
            test_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)

        grouped_evals = [cls_eval, test_eval] 
        pred_gap = self.group_regressor_eval(grouped_evals, disc_eval, self.base_ds,
                                             test_ds, sampler_gen)
        
        regressor_target = self.parameters["regressor_target"]
        regressor_features = self.parameters["regressor_features"]

        pred_acc = cls_eval[regressor_target] - pred_gap
        gap  = cls_eval[regressor_target] - test_eval[regressor_target]
        error = np.abs(gap - pred_gap)
        test_acc = test_eval[regressor_target]
        base_acc = cls_eval[regressor_target]

        p_token = token_maker(cls_eval_token, test_eval_token, cal_token)

        base_ds_name = self.base_ds[0].name
        if self.base_ds[0].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_nam += "_" + test_ds[0].sub_name

        if "regressor_type" not in self.parameters:
            regressor_type = "standard"
        else:
            regressor_type = self.parameters["regressor_type"]

        model_type = self.parameters["regressor_model"]["model_type"]

        cal_alg_name = "mmd_%s_regressor_%s_%s" % (self.mmd_type, regressor_type, model_type)

        int_df = {"base_ds": base_ds_name,
                  "target_ds": target_ds_name,
                  "model": self.model_naming,
                  "base_token": base_token,
                  "token": p_token,
                  "cal_token": cal_token,
                  "cal_alg": cal_alg_name,
                  "gap": gap,
                  "pred_gap": pred_gap,
                  "error": error,
                  "accuracy": test_acc,
                  "pred_acc": pred_acc,
                  "base_acc": base_acc} 

        check  = d_utils.db.insert_into_table("performances", int_df, conn=self.conn)

        return pred_gap, cls_eval[regressor_target]-test_eval[regressor_target], cls_eval, test_eval







class FrechetModelDistance(BaseTrainer):

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming
        self.base_mean = None
        self.base_cov = None
        self.sampler_gen = None
        self.conn = None
    
    def get_base_mean_cov(self):
        if self.base_mean is None:
            base_ds_sampler = self.sampler_gen(self.base_ds[0], self.base_ds[0], [])
            self.base_mean = calc_mean(base_ds_sampler, self.feat_net, self.device)
        if self.base_cov is None:
            base_ds_sampler = self.sampler_gen(self.base_ds[0], self.base_ds[0], [])
            self.base_cov  = calc_cov(base_ds_sampler, self.feat_net, self.base_mean, self.device)

        return self.base_mean, self.base_cov


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None):
        
        # TODO: Handle this more elegantly
        base_ds = base_ds[0]

        # Do this in initialization
        self.feat_net = feat_net
        self.feat_net.to(self.device)

        self.base_ds = base_ds
        self.sampler_gen = sampler_gen

        base_ds_sampler = sampler_gen(base_ds[0], base_ds[0], [])

        disc_evals = []
        cls_evals  = []

        with torch.no_grad():
            for target_ds in train_ds:

                disc_eval_token = token_maker(base_token, "frechet",
                                      base_ds, target_ds,
                                      sampler_gen)

                # PARAMETER NAMINGS
                base_ds_name = self.base_ds[0].name
                if self.base_ds[0].sub_name is not None:
                    base_ds_name += "_" + self.base_ds[0].sub_name

                target_ds_name = target_ds[0].name
                if target_ds[0].sub_name is not None:
                    target_ds_name += "_" + target_ds[0].sub_name

                dist_full = "frechet"


                if d_utils.db.check_existence("distances", {"id":disc_eval_token}, conn=self.conn):
                    logging.info("FOUND IT")
                    disc_eval = d_utils.db.get_selection_table("distances", 
                            match_e={"id":disc_eval_token},
                            conn=self.conn)
                    disc_eval = disc_eval[0]
                    disc_eval["target_ds"] = target_ds_name
                    disc_eval["base_ds"] = base_ds_name
                    disc_eval["model"] = self.model_naming
                    disc_eval["dist_type"] = dist_full
                    check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
                else:
                    logging.info("DIDN'T FIND IT")
                    target_ds_sampler = sampler_gen(target_ds[0], target_ds[0], [])

                    target_mean = calc_mean(target_ds_sampler, feat_net, self.device)
                    target_cov  = calc_cov(target_ds_sampler, feat_net, target_mean, self.device)

                    base_mean, base_cov = self.get_base_mean_cov()
                    mean_dist_sq = torch.pow(base_mean - target_mean, 2).sum(0)

                    logging.info("Base Mean then Base Cov")
                    logging.info(base_mean.shape)
                    logging.info(base_cov.shape)
                    logging.info(target_cov.shape)
                    tr_covmean = torch.trace(torch.sqrt(base_cov * target_cov))
                    trace_dist = torch.trace(base_cov)+ torch.trace(target_cov) - 2*tr_covmean
                    frechet_dist = mean_dist_sq + trace_dist
                    frechet_dist = frechet_dist.squeeze()
                    frechet_dist = frechet_dist.cpu().item()
                    logging.info("Frechet Distance: %f" %frechet_dist)

                    disc_eval = {"score": frechet_dist}
                    disc_eval["id"] =  disc_eval_token
                    disc_eval["target_ds"] = target_ds_name
                    disc_eval["base_ds"] = base_ds_name
                    disc_eval["model"] = self.model_naming
                    disc_eval["dist_type"] = dist_full
                    check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)

                
                disc_evals.append(disc_eval)

                #bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0], target_ds[0])
                bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0], target_ds[0])

                if bf is not None or bc is not None:
                    cls_eval_token_0 = token_maker(base_token, base_ds, sampler_gen,
                                          self.parameters["cls_eval_params"],
                                          target_ds)
                else:
                    cls_eval_token_0 = token_maker(base_token, base_ds, sampler_gen,
                                          self.parameters["cls_eval_params"])

                if d_utils.db.check_existence("classifications", {"id":cls_eval_token_0}, conn=self.conn):

                    logging.info("Found Base Classifier")
                    logging.info("Token: %s" %str(cls_eval_token_0))
                    ce_0 = d_utils.db.get_selection_table("classifications",
                            match_e={"id":cls_eval_token_0}, conn=self.conn)
                    ce_0 = ce_0[0]
                    ce_0["base_ds"] = target_ds_name
                    ce_0["target_ds"] = base_ds_name
                    ce_0["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
                else:
                    logging.info("Computing Base Classifier")
                    ce_0 = eval_classifier(self.feat_net, base_ds_sampler.all,
                              self.parameters["cls_eval_params"],
                              device=self.device,
                              filtering=bf,
                              conversion=bc)
                    
                    ce_0["id"] = cls_eval_token_0
                    ce_0["base_ds"] = target_ds_name
                    ce_0["target_ds"] = base_ds_name
                    ce_0["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
                
                if targetf is not None or tc is not None:
                    cls_eval_token_1 = token_maker(base_token, target_ds, sampler_gen,
                                       self.parameters["cls_eval_params"],
                                       base_ds)
                else:
                    cls_eval_token_1 = token_maker(base_token, target_ds, sampler_gen,
                                       self.parameters["cls_eval_params"])

                #if check_token_ds(cls_eval_token_1, CLS_DF):
                if d_utils.db.check_existence("classifications", {"id":cls_eval_token_1}, conn=self.conn):
                    logging.info("Found Target Classifier")
                    ce_1 = d_utils.db.get_selection_table("classifications",
                            match_e={"id":cls_eval_token_1}, conn=self.conn)
                    ce_1 = ce_1[0]
                    ce_1["base_ds"] = base_ds_name
                    ce_1["target_ds"] = target_ds_name
                    ce_1["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)
                    logging.info(ce_1)
                else:
                    logging.info("Computing Target Classifier")
                    ce_1 = eval_classifier(self.feat_net, target_ds_sampler.all,
                            self.parameters["cls_eval_params"], 
                            device=self.device,
                            filtering=targetf,
                            conversion=tc)

                    ce_1["id"] = cls_eval_token_1
                    ce_1["base_ds"] = base_ds_name
                    ce_1["target_ds"] = target_ds_name
                    ce_1["model"] = self.model_naming
                    check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)

                cls_evals.append([ce_0, ce_1])


        logging.info(disc_evals)
        logging.info(cls_evals)

        logging.info("Class and Disc Evals above")

        regressor_model = networks.DiscWrapper(self.parameters["regressor_model"]["model_type"],
                input_dims=len(self.parameters["regressor_features"]),
                output_dims=self.parameters["regressor_model"]["output_dims"],
                flatten=False)

        regressor_model.to(self.device)
        
        regressor_model, reg_results = self.regressor_training(regressor_model, cls_evals, disc_evals,
                train_ds, sampler_gen)
         

        self.regressor_model = regressor_model

        ## Do Eval Generic stuff
        ##

        base_ds_sampler = sampler_gen(base_ds[0], base_ds[0], []) #, *self.parameters["sampler_params"])
        
        cls_eval_token = token_maker(base_token, base_ds, sampler_gen,
                             self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            cls_eval = d_utils.db.get_selection_table("classifications",
                    match_e={"id": cls_eval_token})
            cls_eval = cls_eval[0]
            self.cls_eval = cls_eval
        else:

            self.cls_eval = eval_classifier(self.feat_net, 
                    base_ds_sampler.all, self.parameters["cls_eval_params"],
                    device=self.device)

            self.cls_eval["id"] = cls_eval_token
            check = d_utils.db.insert_into_table("classifications", self.cls_eval, conn=self.conn)
    

        return reg_results


    def group_prediction(self, test_ds, sampler_gen, base_token=None, cal_token=None):
        """
        Group prediction with frechet distance
        """

        base_ds_sampler = sampler_gen(self.base_ds[0], self.base_ds[0], [test_ds[0]])#, *self.parameters["sampler_params"])
        test_ds_sampler = sampler_gen(test_ds[0], test_ds[0], [self.base_ds[0]]) #, *self.parameters["sampler_params"])


        # PARAMETER NAMINGS
        base_ds_name = self.base_ds[1].name
        if self.base_ds[1].sub_name is not None:
            base_ds_name += "_" + self.base_ds[1].sub_name

        target_ds_name = test_ds[1].name
        if test_ds[1].sub_name is not None:
            target_ds_name += "_" + test_ds[1].sub_name

        dist_full = "frechet"

        #Update to this section complete

        disc_eval_token = token_maker(base_token, "frechet",
                              self.base_ds, test_ds,
                              sampler_gen)

        if d_utils.db.check_existence("distances", {"id": disc_eval_token}, conn=self.conn):
            logging.info("distance found: %s" %disc_eval_token)
            disc_eval = d_utils.db.get_selection_table("distances",
                    match_e={"id": disc_eval_token}, conn=self.conn)
            disc_eval = disc_eval[0]
            disc_eval["base_ds"] = base_ds_name
            disc_eval["target_ds"] = target_ds_name
            disc_eval["model"] = self.model_naming
            disc_eval["dist_type"] = dist_full
            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
        else:
            
            target_mean = calc_mean(test_ds_sampler, self.feat_net, self.device)
            target_cov  = calc_cov(test_ds_sampler, self.feat_net, target_mean, self.device)

            base_mean, base_cov = self.get_base_mean_cov()
            mean_dist_sq = torch.pow(base_mean - target_mean, 2).sum(0)
            tr_covmean = torch.trace(torch.sqrt(base_cov * target_cov))
            trace_dist = torch.trace(base_cov)+ torch.trace(target_cov) - 2*tr_covmean
            frechet_dist = mean_dist_sq + trace_dist
            frechet_dist = frechet_dist.cpu().item()
            logging.info("Frechet Distance: %f" %frechet_dist)

            disc_eval = {"score": frechet_dist}
            disc_eval["id"] = disc_eval_token
            disc_eval["base_ds"] = base_ds_name
            disc_eval["target_ds"] = target_ds_name
            disc_eval["model"] = self.model_naming
            disc_eval["dist_type"] = dist_full
            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)


        
        bf, bc, targetf, tc = get_filters_and_conversions(self.base_ds[1], test_ds[1])

        if bf is not None or bc is not None:
            cls_eval_token = token_maker(base_token, self.base_ds, sampler_gen,
                                 self.parameters["cls_eval_params"],
                                 test_ds)
        else:
            cls_eval_token = token_maker(base_token, self.base_ds, sampler_gen,
                                 self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            logging.info("Found Base Classifier")
            cls_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":cls_eval_token}, conn=self.conn)
            cls_eval = cls_eval[0]
            cls_eval["base_ds"] = target_ds_name
            cls_eval["target_ds"] = base_ds_name
            cls_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)
            logging.info(cls_eval)
        else:
            logging.info(bf)
            logging.info(bc)
            cls_eval = eval_classifier(self.feat_net,
                    base_ds_sampler.all,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=bf,
                    conversion=bc)

            cls_eval["id"] = cls_eval_token
            cls_eval["base_ds"] = target_ds_name
            cls_eval["target_ds"] = base_ds_name
            cls_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)

        if targetf is not None or tc is not None:
            test_eval_token = token_maker(base_token, test_ds, sampler_gen,
                                   self.parameters["cls_eval_params"],
                                   self.base_ds)
        else:
            test_eval_token = token_maker(base_token, test_ds, sampler_gen,
                                   self.parameters["cls_eval_params"])

        if d_utils.db.check_existence("classifications", {"id":test_eval_token}, conn=self.conn):
            logging.info("group prediction test token: " + test_eval_token)
            test_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":test_eval_token})
            test_eval = test_eval[0]
            test_eval["base_ds"] = base_ds_name
            test_eval["target_ds"] = target_ds_name
            test_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)
        else:
            test_eval = eval_classifier(self.feat_net,
                    test_ds_sampler.test,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=targetf,
                    conversion=tc)
            
            test_eval["id"] = test_eval_token
            test_eval["base_ds"] = base_ds_name
            test_eval["target_ds"] = target_ds_name
            test_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)

        grouped_evals = [cls_eval, test_eval] 
        pred_gap = self.group_regressor_eval(grouped_evals, disc_eval, self.base_ds,
                                             test_ds, sampler_gen)
        
        regressor_target = self.parameters["regressor_target"]
        regressor_features = self.parameters["regressor_features"]

        pred_acc = cls_eval[regressor_target] - pred_gap
        gap  = cls_eval[regressor_target] - test_eval[regressor_target]
        error = np.abs(gap - pred_gap)
        test_acc = test_eval[regressor_target]
        base_acc = cls_eval[regressor_target]

        p_token = token_maker(cls_eval_token, test_eval_token, cal_token)

        base_ds_name = self.base_ds[1].name
        if self.base_ds[1].sub_name is not None:
            base_ds_name += "_" + self.base_ds[1].sub_name

        target_ds_name = test_ds[1].name
        if test_ds[1].sub_name is not None:
            target_ds_nam += "_" + test_ds[1].sub_name

        if "regressor_type" not in self.parameters:
            regressor_type = "standard"
        else:
            regressor_type = self.parameters["regressor_type"]

        model_type = self.parameters["regressor_model"]["model_type"]

        cal_alg_name = "frechet_regressor_%s_%s" % (regressor_type, model_type)

        int_df = {"base_ds": base_ds_name,
                  "target_ds": target_ds_name,
                  "model": self.model_naming,
                  "base_token": base_token,
                  "token": p_token,
                  "cal_token": cal_token,
                  "cal_alg": cal_alg_name,
                  "gap": gap,
                  "pred_gap": pred_gap,
                  "error": error,
                  "accuracy": test_acc,
                  "pred_acc": pred_acc,
                  "base_acc": base_acc} 

        check  = d_utils.db.insert_into_table("performances", int_df, conn=self.conn)

        return pred_gap, cls_eval[regressor_target]-test_eval[regressor_target], cls_eval, test_eval

