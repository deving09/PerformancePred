import torch.nn as nn
from torchvision import transforms




config = dict(name="empty",
        parameters = dict(
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
            )
            
        )

