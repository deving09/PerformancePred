import torch
import networks
import logging

from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from .evaluators import *
from .helpers import *
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from .basetrainer import *

from tqdm import tqdm
import numpy as np

import utils as d_utils


def train_temperature(feat_net, train_ds, parameters, sampler_gen,
        device=None, base_token=None, model_naming=None):
    """
    Train temperature scaling model
    """
    
    
    train_ds_0 = [d[0] for d in train_ds]
    train_ds_1 = [d[1] for d in train_ds]
    target_ds_sampler = sampler_gen(train_ds_0, train_ds_1, [])

    epochs = parameters["train_params"]["epochs"]
    lr = parameters["train_params"]["optimizer"]["lr"]
    momentum = parameters["train_params"]["optimizer"]["momentum"]
    
    disc_eval_token = token_maker(base_token, "standard_temp",
                                  train_ds_0, sampler_gen,
                                  parameters["train_params"]
                                  )

    # PARAMETER NAMINGS
    target_ds_name = "_".join([d[0].name for d in train_ds])

    if "temp_type" not in parameters:
        temp_type = "standard"
    else:
        temp_type = parameters["temp_type"]

    dist_full = "temp_%s" % temp_type
    
    if d_utils.db.check_existence("distances", {"id":disc_eval_token}, conn=self.conn):
        logging.info("Reading in Temperature")
        de = d_utils.db.get_selection_table("distances", 
                    match_e={"id":disc_eval_token})
        de = de[0]
        temperature = de["score"]
        
        de["target_ds"] = target_ds_name
        de["model"] = model_naming
        de["dist_type"] = dist_full

        check = d_utils.db.insert_into_table("distances", de, conn=self.conn)
    else:

        temperature = torch.tensor([1.5], requires_grad=True) #, device=device)
        temperature.requires_grad_(True)
        

        logits_list = []
        labels_list = []
        
        logging.info("Starting LBFGS parameter trainings")


        with torch.no_grad():
            for inputs, labels in tqdm(target_ds_sampler.train):
                inputs = inputs.to(device)

                logits = feat_net(inputs)
                logits = logits.cpu()
                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
        

        logging.info(logits.shape)
        logging.info(labels.shape)


        optimizer = optim.LBFGS([temperature],
                                 lr=0.01, max_iter=50)

        def eval():
            scaled_logits = logits/temperature
            loss = nn.functional.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        logging.info("Optimizer for LBFGS started")
        optimizer.step(eval)

        temperature = temperature.to(device)
        
        logging.info("temperature: %f" % temperature.item())
        de = {"score": temperature.item()}
        de["id"] = disc_eval_token
        de["target_ds"] = target_ds_name
        de["model"] = model_naming
        de["dist_type"] = dist_full

        check = d_utils.db.insert_into_table("distances", de)

    return temperature




class TemperatureScalingAlg(BaseTrainer):
    """
    Handle temperature scaling in the same manner of other algorithms
    """

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming=model_naming
        self.conn = conn


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None, batch_norm=False):

        base_ds = base_ds[0]

        self.feat_net = feat_net

        self.feat_net.to(self.device)

        self.base_ds = base_ds

        if "temp_type" not in self.parameters or self.parameters["temp_type"] == "standard":
            logging.info("TRAINING A GLOBAL TEMPERATURE")
            temperature = train_temperature(self.feat_net, train_ds, self.parameters, 
                                            sampler_gen, base_token=base_token,
                                            device=self.device,
                                            model_naming=self.model_naming)

        elif self.parameters["temp_type"] == "val":
            logging.info("TRAINING GLOBAL TEMPERATURE on Val")
            temperature = train_temperature(self.feat_net, [base_ds], self.parameters, 
                                            sampler_gen, base_token=base_token,
                                            device=self.device, 
                                            model_naming=self.model_naming)
        elif self.parameters["temp_type"] == "base_temp":
            logging.info("Setting a GLOBAL TEMPERATURE")
            temperature = 1.0
        else:
            raise ValueError("unknown temperature")

        self.temperature = temperature


        cls_evals = []

        for cls_num, target_ds  in enumerate(train_ds):
            base_ds_sampler = sampler_gen(base_ds[1], base_ds[1], [target_ds[1]])
            target_ds_sampler = sampler_gen(target_ds[1], target_ds[1], [base_ds[1]])


            # Batch Norm Execution
            if batch_norm:
                feat_net.train()
            else:
                feat_net.eval()

            # PARAMETER NAMINGS
            base_ds_name = self.base_ds[0].name
            if self.base_ds[0].sub_name is not None:
                base_ds_name += "_" + self.base_ds[0].sub_name

            target_ds_name = target_ds[0].name
            if target_ds[0].sub_name is not None:
                target_ds_name += "_" + target_ds[0].sub_name


            if "temp_type" not in self.parameters:
                temp_type = "standard"
            else:
                temp_type = self.parameters["temp_type"]

            dist_full = "temp_%s" % temp_type
            
            
            bf, bc, targetf, tc = get_filters_and_conversions(base_ds[1],
                             target_ds[1])

            cls_eval_token_0 = temperature_token_builder(base_token,
                    base_ds, sampler_gen, self.parameters["cls_eval_params"],
                    target_ds,filt=bf, conv=bc, temperature=self.temperature)


            logging.info(self.parameters["cls_eval_params"])
            if d_utils.db.check_existence("classifications", {"id":cls_eval_token_0}, conn=self.conn):
                logging.info("Found Base Classifier")
                ce_0 = d_utils.db.get_selection_table("classifications", 
                            match_e={"id":cls_eval_token_0},
                            conn=self.conn)
                ce_0 = ce_0[0]
                ce_0["base_ds"] = target_ds_name
                ce_0["target_ds"] = base_ds_name
                ce_0["model"] = self.model_naming

                logging.info(ce_0)
                check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
                logging.info(ce_0)
            else:
                logging.info("Computing Base Classifier")
                ce_0 =  eval_classifier(self.feat_net, base_ds_sampler.all,
                                        self.parameters["cls_eval_params"], 
                                        device=self.device,
                                        filtering=bf,
                                        conversion=bc,
                                        temperature=self.temperature)
                ce_0["id"] = cls_eval_token_0
                ce_0["base_ds"] = target_ds_name
                ce_0["target_ds"] = base_ds_name
                ce_0["model"] = self.model_naming
                
                logging.info(ce_0)
                check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
           
            cls_eval_token_1 = temperature_token_builder(base_token,
                    target_ds, sampler_gen, self.parameters["cls_eval_params"],
                    base_ds,filt=targetf, conv=tc, temperature=self.temperature)
            

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
                        self.parameters["cls_eval_params"], device=self.device,
                        filtering=targetf,
                        conversion=tc,
                        temperature=self.temperature)

                ce_1["id"] = cls_eval_token_1
                ce_1["base_ds"] = base_ds_name
                ce_1["target_ds"] = target_ds_name
                ce_1["model"] = self.model_naming
                logging.info(ce_1)
                check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)
                
            cls_evals.append([ce_0, ce_1])

            logging.info("Another Classification Done: %d" %cls_num)

        
        disc_evals = [{"score": temperature}] * len(cls_evals)
        logging.info(disc_evals)
        logging.info(cls_evals)

        regressor_model = networks.DiscWrapper(self.parameters["regressor_model"]["model_type"], 
                           input_dims=len(self.parameters["regressor_features"]),
                           output_dims=self.parameters["regressor_model"]["output_dims"],
                           flatten=False)

        regressor_model, reg_results = self.regressor_training(regressor_model, cls_evals, disc_evals, train_ds, sampler_gen)

        self.regressor_model = regressor_model

        return reg_results



    def group_prediction(self, test_ds, sampler_gen, 
            base_token=None, cal_token=None):
        """
        Group prediction also returns discriminators evals
        """

        base_ds_sampler = sampler_gen(self.base_ds[0], self.base_ds[1], [test_ds[0]])
        test_ds_sampler = sampler_gen(test_ds[0], test_ds[1], [self.base_ds[0]])

        # PARAMETER NAMINGS
        base_ds_name = self.base_ds[0].name
        if self.base_ds[0].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name

        if "temp_type" not in self.parameters:
            temp_type = "standard"
        else:
            temp_type = self.parameters["temp_type"]

        dist_full = "temp_%s" % temp_type
        
        
        bf, bc, targetf, tc = get_filters_and_conversions(self.base_ds[0],
                         test_ds[0])
        
        
        cls_eval_token = temperature_token_builder(base_token,
                self.base_ds, sampler_gen, self.parameters["cls_eval_params"],
                test_ds,filt=bf, conv=bc, 
                temperature=self.temperature)
        

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            logging.info("Found Base Classifier")
            cls_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":cls_eval_token}, conn=self.conn)
            cls_eval = cls_eval[0]
            cls_eval["model"] = self.model_naming
            cls_eval["target_ds"] = base_ds_name
            cls_eval["base_ds"] = target_ds_name
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)
            logging.info(cls_eval)
        else:
            logging.info("Evaluating Base DS: %s" % self.base_ds[0].root)
            cls_eval = eval_classifier(self.feat_net,
                    base_ds_sampler.all, 
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=bf,
                    conversion=bc,
                    temperature=self.temperature)

            cls_eval["id"] = cls_eval_token
            cls_eval["model"] = self.model_naming
            cls_eval["target_ds"] = base_ds_name
            cls_eval["base_ds"] = target_ds_name
            logging.info(cls_eval)
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)
            

        test_eval_token = temperature_token_builder(base_token,
                test_ds, sampler_gen, self.parameters["cls_eval_params"],
                self.base_ds,filt=targetf, conv=tc, 
                temperature=self.temperature)

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name


        if d_utils.db.check_existence("classifications", {"id":test_eval_token}, conn=self.conn):
            logging.info("group prediction test token: " + test_eval_token)
            test_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":test_eval_token}, conn=self.conn)
            test_eval = test_eval[0]
            test_eval["model"] = self.model_naming
            test_eval["target_ds"] = target_ds_name
            test_eval["base_ds"] = base_ds_name
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)
            logging.info(test_eval)
        else:
            logging.info("Evaluating Test DS: %s" % test_ds[0].root)
            test_eval = eval_classifier(self.feat_net,
                    test_ds_sampler.all,
                    self.parameters["cls_eval_params"],
                    temperature=self.temperature,
                    device=self.device,
                    filtering=targetf,
                    conversion=tc)

            test_eval["id"] = test_eval_token
            test_eval["model"] = self.model_naming
            test_eval["target_ds"] = target_ds_name
            test_eval["base_ds"] = base_ds_name
            logging.info(test_eval)
            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)

        
        grouped_cls = [cls_eval, test_eval]
        disc_eval = {"dummy": 0.0}

        pred_gap = self.group_regressor_eval(grouped_cls, disc_eval,
                self.base_ds, test_ds, sampler_gen)

        regressor_target = self.parameters["regressor_target"]
        

        pred = cls_eval[regressor_target]- pred_gap
        logging.info("Pred Accuracy: %f" % pred)
        logging.info("Test Accuracy: %f" % test_eval[regressor_target])
        
        base_ds_name = self.base_ds[0].name
        if self.base_ds[0].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name

        if "temp_type" not in self.parameters:
            temp_type = "standard"
        else:
            temp_type = self.parameters["temp_type"]

        if "regressor_type" not in self.parameters:
            regressor_type = "standard"
        else:
            regressor_type = self.parameters["regressor_type"]
        
        reg_feats = "-".join(self.parameters["regressor_features"])
        model_type = self.parameters["regressor_model"]["model_type"]
        cal_alg_name = "temp_%s_regressor_%s_%s_feat_%s" % (temp_type, 
                regressor_type, model_type, reg_feats)

        gap = cls_eval[regressor_target] - test_eval[regressor_target]
        error = np.abs(gap - pred_gap)
        test_acc = test_eval[regressor_target]
        base_acc = cls_eval[regressor_target]

        p_token = token_maker(cls_eval_token, test_eval_token, cal_token)
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
                  "pred_acc": pred,
                  "base_acc": base_acc}
        
        check  = d_utils.db.insert_into_table("performances", int_df, conn=self.conn)
        logging.info(int_df)
        return pred_gap, cls_eval[regressor_target] - test_eval[regressor_target], cls_eval, test_eval

