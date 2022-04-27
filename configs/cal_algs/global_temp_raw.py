import torch.nn as nn
from torchvision import transforms




config = dict(name="temperature",
        parameters = dict(
            temp_type = "standard",
            model_params = dict(
                model_type  = "linear",
                output_dims = 2,
                hidden_size = None,
                flatten     = None
                ),
            train_params = dict(
                epochs = 10,
                criterion = nn.functional.cross_entropy,
                scheduler = dict(
                    scheduler_type = "step_lr",
                    step_size      = 100,
                    gamma          = 0.1
                    ),
                optimizer = dict(
                    opt_wt   = "disc_opt",
                    opt      = "sgd",
                    momentum = 0.9,
                    wd       = 0.0001,
                    lr       = 0.001
                    )
            ),
            eval_params = [("accuracy", None),
                ("auc", None),
                ("brier", None),
                ("ece", None),
                ("test_avg", None),
                ("loss", None)],
            cls_eval_params = [("accuracy", None),
                ("loss", None),
                ("test_avg", None),
                ("entropy", None)],
            regressor_type = "acc_raw",
            regressor_model = dict(
                model_type  = "linear",
                #input_dims = None,
                output_dims = 1, 
                hidden_size = None,
                flatten     = None
                ),
            regressor_features = ["test_avg"],
            regressor_target   = "accuracy",
            regressor_params   = dict(
                epochs = 5,
                criterion = nn.functional.cross_entropy,
                scheduler = dict(
                    scheduler_type = "step_lr",
                    step_size      = 10000,
                    gamma          = 0.1
                    ),
                optimizer = dict(
                    opt_wt   = "disc_opt",
                    opt      = "sgd",
                    momentum = 0.9,
                    wd       = 0.0001,
                    lr       = 0.001
                    )
                
                ),
            regressor_eval_params=[("r2_score", None),
                    ("mse", None),
                    ("mae", None)]
            )
            
        )

