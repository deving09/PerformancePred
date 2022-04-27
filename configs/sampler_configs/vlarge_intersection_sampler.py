import torch.nn as nn
from torchvision import transforms


config = dict(name = "intersection",
                sampler_params=dict(
                    batch_size=1024,
                    random_seed=73,
                    num_workers=16)
             )


