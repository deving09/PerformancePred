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


class PretextCalibrator(BaseTrainer):
    """
    Pretext calibrator we will start with rotation 
    """

    def __init__(self, parameters, device=None, model_naming=None, conn=None):
        self.parameters = parameters
        self.feat_net = None
        self.device = device
        self.model_naming = model_naming
        self.conn = conn


    def train(self, feat_net, base_ds, train_ds, sampler_gen, base_token=None):
        base_ds = base_ds[0]

        # Initialize feat net
        self.feat_net = feat_net
        self.feat_net.to(self.device)

        logging.info(self.feat_net.dims)
        self.base_ds = base_ds

        disc_evals = []
        cls_evals  = []

        self.disc_model = None
        
        for target_ds in train_ds:
            target_ds_sampler = sampler_gen(target_ds[0],target_ds[1], [self.base_ds[0]])

            bf, bc, targetf, tc = get_filters_and_conversions(base_ds[0],
                                       target_ds[0])

            
            disc_eval_token = token_maker(base_token, 
                                         "rotation_pretext",
                                         base_ds, target_ds,
                                         sampler_gen, 
                                         self.parameters["model_params"],
                                         self.parameters["train_params"],
                                         self.parameters["eval_params"],
                                         build_pretext_labels)

            base_ds_name = base_ds[0].name

            if base_ds[0].sub_name is not None:
                base_ds_name += "_" + base_ds[0].sub_name

            target_ds_name = target_ds[0].name
            if target_ds[0].sub_name is not None:
                target_ds_name += "_" + target_ds[0].sub_name


            if d_utils.db.check_existence("distances", {"id":disc_eval_token}, conn=self.conn):
                logging.info("Found Distance for Pretext")
                disc_eval = d_utils.db.get_selection_table("distances",
                        match_e={"id":disc_eval_token},
                        conn=self.conn)

                disc_eval = disc_eval[0]
                disc_eval["target_ds"] = target_ds_name
                disc_eval["base_ds"] = base_ds_name
                disc_eval["dist_type"] = "rotation_pretext"
                disc_eval["model"] = self.model_naming

                check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
                disc_evals.append(disc_eval)
            else:

                if self.disc_model is None:
                    logging.info("Train Disc Model")
                    curr_model = networks.DiscWrapper(self.parameters["model_params"]["model_type"], 
                            input_dims=self.feat_net.dims,
                            output_dims=4)

                    curr_model.to(self.device)
                    
                    base_ds_sampler = sampler_gen(base_ds[0], base_ds[1], [])

                    new_feat_net, curr_model = train_model(self.feat_net, curr_model,
                            [base_ds_sampler],
                            self.parameters["train_params"],
                            device=self.device,
                            build_labels=build_pretext_labels,
                            base_ds_name=base_ds_name,
                            target_ds_name="rotation_pretext")

                    self.disc_model = curr_model

                logging.info("Didn't find Distance for Pretext")
            
                def disc_pass(x):
                    x = new_feat_net.disc_forward(x)
                    x = curr_model(x)
                    return x

                disc_eval = eval_discriminator(disc_pass,
                    [target_ds_sampler.all],
                    self.parameters["eval_params"],
                    device=self.device,
                    build_labels=build_pretext_labels)
                
                disc_eval["id"] = disc_eval_token
                disc_eval["target_ds"] = target_ds_name
                disc_eval["base_ds"] = base_ds_name
                disc_eval["dist_type"] = "rotation_pretext"
                disc_eval["model"] = self.model_naming

                check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
                disc_evals.append(disc_eval)


            cls_eval_token_0 = temperature_token_builder(
                    base_token, base_ds,
                    sampler_gen, self.parameters["cls_eval_params"],
                    target_ds, filt=bf, conv=bc)

            cls_eval_token_1 = temperature_token_builder(
                    base_token, target_ds, 
                    sampler_gen, self.parameters["cls_eval_params"],
                    base_ds, filt=targetf, conv=tc)

            if d_utils.db.check_existence("classifications", {"id": cls_eval_token_0}, conn=self.conn):
                logging.info("Found Base Classifier")
                logging.info("Token: %s" % str(cls_eval_token_0))

                ce_0 = d_utils.db.get_selection_table("classifications",
                        match_e={"id": cls_eval_token_0},
                        conn=self.conn)

                ce_0 = ce_0[0]
                ce_0["base_ds"] = target_ds_name
                ce_0["target_ds"] = base_ds_name
                ce_0["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)

            else:
                logging.info("Computing Base Classifier")
                logging.info("Token: %s" % str(cls_eval_token_0))


                ce_0 =  eval_classifier(self.feat_net,
                    base_ds_sampler.all,
                    self.parameters["cls_eval_params"], device=self.device,
                    filtering=bf, conversion=bc)

                ce_0["id"] = cls_eval_token_0
                ce_0["base_ds"] = target_ds_name
                ce_0["target_ds"] = base_ds_name
                ce_0["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_0, conn=self.conn)


            if d_utils.db.check_existence("classifications", {"id": cls_eval_token_1}, conn=self.conn):
                logging.info("Found Target Classifier")

                ce_1 = d_utils.db.get_selection_table("classifications",
                        match_e={"id": cls_eval_token_1},
                        conn=self.conn)
                ce_1 = ce_1[0]
                ce_1["base_ds"] = base_ds_name
                ce_1["target_ds"] = target_ds_name
                ce_1["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)
                logging.info(ce_1)
            else:
                logging.info("Computing Target Classifier")
                logging.info("Token: %s" % str(cls_eval_token_1))

                ce_1 = eval_classifier(self.feat_net, target_ds_sampler.all,
                        self.parameters["cls_eval_params"], device=self.device,
                        filtering=targetf, convervsion=tc)

                ce_1["id"] = cls_eval_token_1
                ce_1["base_ds"] = base_ds_name
                ce_1["target_ds"] = target_ds_name
                ce_1["model"] = self.model_naming
                check = d_utils.db.insert_into_table("classifications", ce_1, conn=self.conn)
                
                
            cls_evals.append([ce_0, ce_1])

        logging.info(disc_evals)
        logging.info(cls_evals)

        logging.info("New distributions done")

        regressor_model = networks.DiscWrapper(self.parameters["regressor_model"]["model_type"],
                input_dims=len(self.parameters["regressor_features"]),
                output_dims=self.parameters["regressor_model"]["output_dims"],
                flatten=False)

        regressor_model, reg_results = self.regressor_training(regressor_model,
                          cls_evals, disc_evals, train_ds, sampler_gen)

        self.regressor_model = regressor_model

        return reg_results


    def instance_prediction(self, x, cls_eval=None):
        """
        Instance prediction
        """
        pass

    def group_prediction(self, test_ds, sampler_gen, base_token=None, cal_token=None):
        """
        Group prediction
        """

        test_ds_sampler = sampler_gen(test_ds[0], test_ds[1], [self.base_ds[0]])
        base_ds_sampler = sampler_gen(self.base_ds[0], self.base_ds[1], [test_ds[0]])

        # PARAMETER NAMINGS

        base_ds_name = self.base_ds[0].name
        if self.base_ds[0].sub_name is not None:
            base_ds_name += "_" + self.base_ds[0].sub_name

        target_ds_name = test_ds[0].name
        if test_ds[0].sub_name is not None:
            target_ds_name += "_" + test_ds[0].sub_name

        if "regressor_type" not in self.parameters:
            regressor_type = "standard"
        else:
            regressor_type = self.parameters["regressor_type"]


        reg_feats = "-".join(self.parameters["regressor_features"])
        model_type = self.parameters["regressor_model"]["model_type"]

        cal_alg_name = "pretext_rotation_regressor_%s_%s_feat_%s" % (regressor_type, 
                model_type, reg_feats)


        # Do Caching Shit

        disc_eval_token = token_maker(base_token, 
                                     "rotation_pretext",
                                     self.base_ds, test_ds,
                                     sampler_gen, 
                                     self.parameters["model_params"],
                                     self.parameters["train_params"],
                                     self.parameters["eval_params"],
                                     build_pretext_labels)

        if d_utils.db.check_existence("distances", {"id": disc_eval_token}, conn=self.conn):
            logging.info("Found Distances")
            disc_eval = d_utils.db.get_selection_table("distances",
                    match_e={"id": disc_eval_token},
                    conn=self.conn)

            disc_eval = disc_eval[0]
            disc_eval["target_ds"] = target_ds_name
            disc_eval["base_ds"] = base_ds_name
            disc_eval["dist_type"] = "rotation_pretext"
            disc_eval["model"] = self.model_naming

            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)
        else:
            logging.info("Compute Distances")

            if self.disc_model is None:
                logging.info("Train Disc Model")
                base_ds_sampler = sampler_gen(self.base_ds[0], self.base_ds[1], [])
                
                curr_model = networks.DiscWrapper(self.parameters["model_params"]["model_type"], 
                        input_dims=self.feat_net.dims,
                        output_dims=4)

                curr_model.to(self.device)
                

                new_feat_net, curr_model = train_model(self.feat_net, curr_model,
                        [base_ds_sampler],
                        self.parameters["train_params"],
                        device=self.device,
                        build_labels=build_pretext_labels,
                        base_ds_name=base_ds_name,
                        target_ds_name="rotation_pretext")

                self.disc_model = curr_model
            
            
            def disc_pass(x):
                x = new_feat_net.disc_forward(x)
                x = self.disc_model(x)
                return x

            disc_eval = eval_discriminator(disc_pass,
                    [test_ds_sampler.all],
                    self.parameters["eval_params"],
                    device=self.device,
                    build_labels=build_pretext_labels)

            disc_eval["id"]  = disc_eval_token
            disc_eval["target_ds"] = target_ds_name
            disc_eval["base_ds"] = base_ds_name
            disc_eval["dist_type"] = "rotation_pretext"
            disc_eval["model"] = self.model_naming

            check = d_utils.db.insert_into_table("distances", disc_eval, conn=self.conn)

        logging.info(disc_eval)

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
            logging.info("Group predicition test token: " + test_eval_token)

            test_eval = d_utils.db.get_selection_table("classifications",
                    match_e={"id": test_eval_token},
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
        gap = cls_eval[regressor_target] - test_eval[regressor_target]
        error = np.abs(gap- pred_gap)
        test_acc = test_eval[regressor_target]
        base_acc = cls_eval[regressor_target]

        p_token = token_maker(cls_eval_token, test_eval_token, cal_token)

        logging.info("Predicted Acc: %f" % pred_acc)
        logging.info("Actual Acc: %f" % test_acc)

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

        check = d_utils.db.insert_into_table("performances", int_df, conn=self.conn)

        return pred_gap, cls_eval[regressor_target] - test_eval[regressor_target], cls_eval, test_eval

