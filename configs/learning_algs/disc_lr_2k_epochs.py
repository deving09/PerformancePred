import torch.nn as nn
from torchvision import transforms

config = dict(
            epochs = 2000,
            criterion = nn.functional.cross_entropy,
            scheduler = dict(
                scheduler_type = "step_lr",
                step_size      = 500,
                gamma          = 0.1
            ),
            optimizer = dict(
                opt_wt   = "full_opt", #"full_opt", #"disc_opt",
                graded_discount = 0.1,
                opt      = "adam",
                momentum = 0.9,
                wd       = 0.005, #0.005, #0.0001,
                lr       = 0.0001
            )
        )


