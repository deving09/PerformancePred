import torch
import networks
import logging

from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
# Figure out how to handle device
from .evaluators import *
from .helpers import *
from .basetrainer import *
from torch.utils.data import DataLoader, TensorDataset

import utils as d_utils

from tqdm import tqdm
import torch.nn as nn

import numpy as np
import wandb


class EmptyTrainer(object):
    """
    Empty trainer
    """

    def __init__(self):
        pass

    def train(self, feat_net, train_ds, val_ds, sampler_gen, batch_norm=False):
        results = {} 
        return feat_net, results
        #pass


def standard_discriminator_training(feat_net, base_ds, train_ds, parameters, sampler_gen, base_token=None,
        device=None, model_naming=None, conn=None, batch_norm=True):
    models = []
    disc_evals = []
    
    for target_ds in train_ds:

        disc_eval_token =   token_maker(base_token, "standard_disc", 
                                        base_ds, target_ds, 
                                        #self.parameters)
                                        sampler_gen, parameters["model_params"],
                                        parameters["train_params"],
                                        parameters["eval_params"])
        

        base_ds_name = base_ds[0].name
        if base_ds[0].sub_name is not None:
            base_ds_name += "_" + base_ds[0].sub_name

        target_ds_name = target_ds[0].name
        if target_ds[0].sub_name is not None:
            target_ds_name += "_" + target_ds[0].sub_name


        if "disc_type" not in parameters:
            disc_type = "standard"
        else:
            disc_type = parameters["disc_type"]


        dist_type_full = "disc_%s" % (disc_type)

        if d_utils.db.check_existence("distances", {"id": disc_eval_token}, conn=conn):
            logging.info("FOUND IT")
            disc_eval = d_utils.db.get_selection_table("distances", 
                                                match_e={"id":disc_eval_token},
                                                conn=conn)
            disc_eval = disc_eval[0]
            disc_eval["target_ds"] = target_ds_name
            disc_eval["base_ds"] = base_ds_name
            disc_eval["dist_type"] = dist_type_full
            disc_eval["model"] = model_naming

            check = d_utils.db.insert_into_table("distances", disc_eval, conn=conn)
            disc_evals.append(disc_eval)
        else:
            logging.info("DIDN'T FIND IT")

            curr_model = networks.DiscWrapper(parameters["model_params"]["model_type"], input_dims=feat_net.dims, output_dims=parameters["model_params"]["output_dims"])

            logging.info("Feature Dims: %d" % feat_net.dims)
            logging.info("Disc type: %s" % parameters["model_params"]["model_type"])
            curr_model.to(device)

            base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], [target_ds[0]]) 
            #, *self.parameters["sampler_params"])
            target_ds_sampler = sampler_gen(target_ds[0], target_ds[1], [base_ds[0]]) 
            #, *self.parameters["sampler_params"])

            #Update to this section complete
            new_feat_net, curr_model = train_model(feat_net, curr_model, 
                    [base_ds_sampler, target_ds_sampler], 
                    parameters["train_params"], device=device,
                    build_labels=build_disc_labels,
                    base_ds_name=base_ds_name, 
                    target_ds_name=target_ds_name,
                    batch_norm=batch_norm)
           
            model_fn = os.path.join("models", "%s_%s_%s_%s_%s.pth" %(base_ds_name, 
                target_ds_name, dist_type_full, model_naming, disc_eval_token))
            
            feat_model_fn = os.path.join("models", "feat_%s_%s_%s_%s_%s.pth" %(base_ds_name, 
                target_ds_name, dist_type_full, model_naming, disc_eval_token))

            torch.save({"model_state_dict": curr_model.state_dict()}, model_fn)
            
            torch.save({"model_state_dict": new_feat_net.state_dict()}, feat_model_fn)

            logging.info("One model done")
            models.append(curr_model)
            
            # Update to eval discriminators: Complete
            def disc_pass(x):#, feat_net=feat_net, curr_model=curr_model):
                x = new_feat_net.disc_forward(x)
                x = curr_model(x)
                return x
            
            de = eval_discriminator(disc_pass, 
                [base_ds_sampler.test, target_ds_sampler.test], 
                parameters["eval_params"], device=device,
                build_labels=build_disc_labels)
            
            # Saving to DF for future use
            de["id"] = disc_eval_token
            de["target_ds"] = target_ds_name
            de["base_ds"] = base_ds_name
            de["dist_type"] = dist_type_full
            de["model"] = model_naming
            check = d_utils.db.insert_into_table("distances", de, conn=conn)

            disc_evals.append(de)
            

    return disc_evals, models



class BinaryDiscriminator(BaseTrainer):
    """
    Binary Discriminator for Calibration
    """

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming
        self.conn = conn


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None, batch_norm=False):
       
        # Fix this asap
        base_ds = base_ds[0]
        
        # Do this in initialization
        self.feat_net = feat_net

        self.feat_net.to(self.device)
       
        logging.info(self.feat_net.dims)
        self.base_ds = base_ds

        
        if "disc_type" not in self.parameters or self.parameters["disc_type"] == "standard":

            disc_evals, models = standard_discriminator_training(self.feat_net, base_ds, train_ds, 
                                                                 self.parameters, sampler_gen,
                                                                 base_token=base_token,
                                                                 device=self.device,
                                                                 model_naming=self.model_naming,
                                                                 conn=self.conn,
                                                                 batch_norm=batch_norm)
        elif self.parameters["disc_type"] == "grouped":
            logging.info("doing Grouped Shit")
            disc_evals, models = grouped_discriminator_training(self.feat_net, base_ds, train_ds,
                                                                self.parameters, sampler_gen, device=self.device,
                                                                model_naming=self.model_naming)
        else:
            raise NotImplementedError()


        self.models = models
        cls_evals = []

        #for target_ds in train_ds[1:]:
        for target_ds in train_ds:
            base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], [target_ds[0]]) 
            target_ds_sampler = sampler_gen(target_ds[0], target_ds[1],[base_ds[0]]) 

            # PARAMETER NAMINGS
            base_ds_name = self.base_ds[0].name
            if self.base_ds[0].sub_name is not None:
                base_ds_name += "_" + self.base_ds[0].sub_name

            target_ds_name = target_ds[0].name
            if target_ds[0].sub_name is not None:
                target_ds_name += "_" + target_ds[0].sub_name

            if "regressor_type" not in self.parameters:
                regressor_type = "standard"
            else:
                regressor_type = self.parameters["regressor_type"]


            if "disc_type" not in self.parameters:
                disc_type = "standard"
            else:
                disc_type = self.parameters["disc_type"]


            reg_feats = "-".join(self.parameters["regressor_features"])
            model_type = self.parameters["regressor_model"]["model_type"]
            dist_type_full = "disc_%s" % disc_type
            
            #Seperate this
            bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0],
                    target_ds[0])

            cls_eval_token_0 = temperature_token_builder(base_token,
                    base_ds, sampler_gen, self.parameters["cls_eval_params"],
                    target_ds, filt=bf, conv=bc)

            cls_eval_token_1 = temperature_token_builder(base_token,
                    target_ds, sampler_gen, self.parameters["cls_eval_params"],
                    base_ds, filt=targetf, conv=tc)

            
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
                logging.info("Token: %s" %str(cls_eval_token_0))
                ce_0 = eval_classifier(self.feat_net, base_ds_sampler.all,
                        self.parameters["cls_eval_params"], device=self.device,
                        filtering=bf, conversion=bc)

                ce_0["id"] = cls_eval_token_0
                ce_0["base_ds"] = target_ds_name
                ce_0["target_ds"] = base_ds_name
                ce_0["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)
            
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
                logging.info("Token: %s" %str(cls_eval_token_1))
                ce_1 = eval_classifier(self.feat_net, target_ds_sampler.all,
                        self.parameters["cls_eval_params"], device=self.device,
                        filtering=targetf,conversion=tc)
                ce_1["id"] = cls_eval_token_1
                ce_1["base_ds"] = base_ds_name
                ce_1["target_ds"] = target_ds_name
                ce_1["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)

            cls_evals.append([ce_0, ce_1])

            logging.info("Classifer Num: %d" %(len(cls_evals)))
            logging.info(cls_evals[-1])


        logging.info(disc_evals)
        logging.info(cls_evals)
        
        regressor_model = networks.DiscWrapper(self.parameters["regressor_model"]["model_type"],
                input_dims=len(self.parameters["regressor_features"]),
                output_dims=self.parameters["regressor_model"]["output_dims"],
                flatten=False)

        # This will run anytime
        regressor_model, reg_results = self.regressor_training(regressor_model, cls_evals, disc_evals,
                          train_ds, sampler_gen)

        self.regressor_model = regressor_model

        ## Do Eval Generic stuff
        ##

        base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], []) #, *self.parameters["sampler_params"])

        # CACHE THIS


        cls_eval_token = token_maker(base_token, base_ds, sampler_gen,
                                        self.parameters["cls_eval_params"])
        
        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            logging.info("Found Base Classifier")
            cls_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":cls_eval_token}, conn=self.conn)
            cls_eval = cls_eval[0]
            self.cls_eval = cls_eval
        else:
            self.cls_eval = eval_classifier(self.feat_net, 
                    base_ds_sampler.all, self.parameters["cls_eval_params"],
                    device=self.device)
            self.cls_eval["id"] = cls_eval_token
            check = d_utils.db.insert_into_table("classifications", self.cls_eval, conn=self.conn) 


        self.base_temperature = torch.exp(torch.tensor(self.cls_eval["test_avg"] - self.cls_eval["accuracy"]))

        logging.info("Base Temperature: %f" %(self.base_temperature))

        return reg_results

    def instance_prediction(self, x, cls_eval=None):
        """
        Instance prediction
        """

        if cls_eval == None:
            cls_eval = self.cls_eval

        y = self.feat_net(x)
        y_t = y / torch.exp(cls_eval["test_avg"]  - cls_eval["accuracy"])

        return nn.functional.softmax(y_t)


    def group_prediction(self, test_ds, sampler_gen, base_token=None, cal_token=None):
        """
        Group prediction also return discriminators evals
        """

        base_ds_sampler = sampler_gen(self.base_ds[1], self.base_ds[1],[test_ds[1]])#, *self.parameters["sampler_params"])
        test_ds_sampler = sampler_gen(test_ds[1], test_ds[1], [self.base_ds[1]]) #, *self.parameters["sampler_params"])

        # PARAMETER NAMINGS
        base_ds_name = self.base_ds[1].name
        if self.base_ds[1].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[1].name
        if test_ds[1].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name

        if "regressor_type" not in self.parameters:
            regressor_type = "standard"
        else:
            regressor_type = self.parameters["regressor_type"]


        if "disc_type" not in self.parameters:
            disc_type = "standard"
        else:
            disc_type = self.parameters["disc_type"]


        reg_feats = "-".join(self.parameters["regressor_features"])
        model_type = self.parameters["regressor_model"]["model_type"]
        dist_type_full = "disc_%s" % disc_type
        cal_alg_name = "disc_%s_regressor_%s_%s_feat_%s" % (disc_type, regressor_type, 
                model_type, reg_feats)


        # Do Caching Shit

        if "disc_type" not in self.parameters or self.parameters["disc_type"] == "standard":
            disc_type = "standard_disc"
        else:
            disc_type = "group_disc"

        disc_eval_token =   token_maker(base_token, disc_type, 
                                        self.base_ds, test_ds, 
                                        #self.parameters)
                                        sampler_gen, self.parameters["model_params"],
                                        self.parameters["train_params"],
                                        self.parameters["eval_params"])

        target_ds_name = test_ds[0].name                                              
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name
        

        if d_utils.db.check_existence("distances", {"id":disc_eval_token}):
            disc_eval = d_utils.db.get_selection_table("distances",
                         match_e={"id":disc_eval_token}, conn=self.conn)
            disc_eval = disc_eval[0]

            disc_eval["target_ds"] = target_ds_name
            disc_eval["base_ds"] = base_ds_name
            disc_eval["dist_type"] = dist_type_full
            disc_eval["model"] = self.model_naming

            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
        else:

            if "disc_type" not in self.parameters or self.parameters["disc_type"] == "standard":
                curr_model = networks.DiscWrapper(self.parameters["model_params"]["model_type"], 
                        input_dims=self.feat_net.dims, 
                        output_dims=self.parameters["model_params"]["output_dims"])

                curr_model.to(self.device)
                
                #Update to this section complete
                new_feat_net, curr_model = train_model(self.feat_net, curr_model, 
                        [base_ds_sampler, test_ds_sampler], 
                        self.parameters["train_params"], device=self.device,
                        build_labels=build_disc_labels,
                        base_ds_name=base_ds_name,
                        target_ds_name=target_ds_name)
            
                model_fn = os.path.join("models", "%s_%s_%s_%s_%s.pth" %(base_ds_name, 
                    target_ds_name, dist_type_full, self.model_naming, disc_eval_token))

                feat_model_fn = os.path.join("models", "feat_%s_%s_%s_%s_%s.pth" %(base_ds_name, 
                    target_ds_name, dist_type_full, self.model_naming, disc_eval_token))
                torch.save({"model_state_dict": curr_model.state_dict()}, model_fn)
                
                torch.save({"model_state_dict": new_feat_net.state_dict()}, feat_model_fn)
            elif self.parameters["disc_type"] == "grouped":
                logging.info("doing Grouped Shit")
                curr_model = self.models[0]
            else:
                raise NotImplementedError()
            
            
            # Update to eval discriminators: Complete
            def disc_pass(x):
                x = self.feat_net.disc_forward(x)
                x = curr_model(x)
                return x
            
            disc_eval = eval_discriminator(disc_pass, 
                [base_ds_sampler.test, test_ds_sampler.test], 
                self.parameters["eval_params"],
                device=self.device,
                build_labels=build_disc_labels)

            # Saving to DF for future use
            disc_eval["id"] = disc_eval_token
            disc_eval["target_ds"] = target_ds_name
            disc_eval["base_ds"] = base_ds_name
            disc_eval["dist_type"] = dist_type_full
            disc_eval["model"] = self.model_naming
            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)

        
        bf, bc, targetf, tc = get_filters_and_conversions(self.base_ds[0],
                test_ds[0])

        cls_eval_token = temperature_token_builder(base_token,
                self.base_ds, sampler_gen, self.parameters["cls_eval_params"],
                test_ds, filt=bf, conv=bc)
        
        

        if d_utils.db.check_existence("classifications", {"id":cls_eval_token}, conn=self.conn):
            logging.info("Found Base Classifier")
            cls_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":cls_eval_token}, 
                        conn=self.conn)
            cls_eval = cls_eval[0]
            cls_eval["model"] = self.model_naming
            cls_eval["target_ds"] = base_ds_name
            cls_eval["base_ds"] = target_ds_name

            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)
            logging.info(cls_eval)
        else: 
            cls_eval = eval_classifier(self.feat_net,
                    base_ds_sampler.all,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=bf,
                    conversion=bc)
            cls_eval["id"] = cls_eval_token
            cls_eval["model"] = self.model_naming
            cls_eval["target_ds"] = base_ds_name
            cls_eval["base_ds"] = target_ds_name
            check = d_utils.db.insert_into_table("classifications", cls_eval, conn=self.conn)

        regressor_features = self.parameters["regressor_features"]
        regressor_target = self.parameters["regressor_target"]
        
        # Cache Shit
        test_eval_token = temperature_token_builder(base_token,
                test_ds, sampler_gen, self.parameters["cls_eval_params"],
                self.base_ds, filt=targetf, conv=tc)
        
        # Classification on Target
        if d_utils.db.check_existence("classifications", {"id":test_eval_token}, conn=self.conn):
            logging.info("group prediction test token: " + test_eval_token)
            test_eval = d_utils.db.get_selection_table("classifications", 
                        match_e={"id":test_eval_token},
                        conn=self.conn)
            test_eval = test_eval[0]
            test_eval["model"] = self.model_naming
            test_eval["target_ds"] = target_ds_name
            test_eval["base_ds"] = base_ds_name

            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)
        else:
            test_eval = eval_classifier(self.feat_net,
                    test_ds_sampler.all,
                    self.parameters["cls_eval_params"],
                    device=self.device,
                    filtering=targetf,
                    conversion=tc)
            test_eval["id"] = test_eval_token
            test_eval["model"] = self.model_naming
            test_eval["target_ds"] = target_ds_name
            test_eval["base_ds"] = base_ds_name

            check = d_utils.db.insert_into_table("classifications", test_eval, conn=self.conn)

        
        grouped_cls = [cls_eval, test_eval]

        pred_gap = self.group_regressor_eval(grouped_cls, disc_eval,
                      self.base_ds, test_ds, sampler_gen) 
        
        pred_acc = cls_eval[regressor_target] - pred_gap
        gap  = cls_eval[regressor_target] - test_eval[regressor_target]
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
                  "pred_acc": pred_acc,
                  "base_acc": base_acc} 

        check  = d_utils.db.insert_into_table("performances", int_df, conn=self.conn)

        return pred_gap, cls_eval[regressor_target]-test_eval[regressor_target], cls_eval, test_eval
