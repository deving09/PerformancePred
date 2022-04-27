import torch.nn as nn
from torchvision import transforms

config = dict(name="binary_discriminator",
        parameters = dict(
            model_params = dict(
                model_type  = "resclass",
                output_dims = 2,
                hidden_size = None,
                flatten     = None
                ),
            train_params = dict(
                epochs = 2000,
                criterion = nn.functional.cross_entropy,
                scheduler = dict(
                    scheduler_type = "step_lr",
                    step_size      = 500,
                    gamma          = 0.1
                    ),
                optimizer = dict(
                    opt_wt   = "disc_opt",
                    opt      = "adam",
                    momentum = 0.9,
                    wd       = 0.005, #0.0001,
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
            regressor_type = "standard",
            regressor_model = dict(
                model_type  = "linear",
                #input_dims = None,
                output_dims = 1, 
                hidden_size = None,
                flatten     = None
                ),
            regressor_features = ["auc"],
            regressor_target   = "accuracy",
            regressor_params   = dict(
                epochs = 5000,
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
                    lr       = 0.00001
                    )
                
                ),
            regressor_eval_params=[("r2_score", None),
                    ("mse", None),
                    ("mae", None)]
            )
            
        )



