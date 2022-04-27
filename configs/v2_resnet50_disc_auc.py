import torch.nn as nn
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

_dataset_config = dict(
        base = [("imagenet", transform, valid_transform, None)],
        cal  = [("imagenetc-digital-jpeg", transform, valid_transform, None),
         ("imagenetc-weather-frost", transform, valid_transform, None),
         #("imagenetv2-matched", transform, None),
         ("imagenetc-digital-elastic", transform, valid_transform, None),
         ("imagenetc-digital-contrast", transform, valid_transform, None),
         ("imagenetc-digital-pixelate", transform, valid_transform, None),
         ("imagenetc-gaussian-noise", transform, valid_transform, None),
         ("imagenetc-impulse-noise", transform, valid_transform, None),
         ("imagenetc-shot-noise", transform,valid_transform,  None),
         ("imagenetc-weather-fog", transform, valid_transform, None),
         ("imagenetc-weather-snow", transform, valid_transform, None),
         ("imagenetc-weather-brightness", transform, valid_transform, None)],
        test = [ #("imagenetv2-top", transform, None),
         ("imagenetv2-top-fixed", transform, valid_transform, None),
         ("imagenetv2-matched-fixed", transform, valid_transform, None),
         ("imagenetv2-threshold-fixed", transform, valid_transform, None),
         ("imagenetc-motion-blur", transform, valid_transform, None),
         ("imagenetc-defocus-blur", transform, valid_transform, None),
         ("imagenetc-glass-blur", transform, valid_transform, None),
         ("imagenetc-zoom-blur", transform, valid_transform, None),
         ("imagenetc-extra-gaussian", transform, valid_transform, None),
         ("imagenetc-extra-speckle", transform, valid_transform, None),
         ("imagenetc-extra-spatter", transform, valid_transform, None),
         ("imagenetc-extra-saturate", transform, valid_transform, None)],
 )

_model_config = dict(
        net =  "resnet50",
        layer_probe = "penultimate",
        pretrained = True
)

# How to do this
_base_train_algorithm = "EmptyTrainer"

_sampler = dict(name = "standard")

_cal_algorithm = dict(name="binary_discriminator",
        parameters = dict(
            temp_type = "base_temp",
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
                    lr       = 0.00001
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
                ("test_avg", None)],
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

config = dict(
        dataset = _dataset_config,
        model   = _model_config,
        base_train_algorithm = _base_train_algorithm,
        cal_algorithm = _cal_algorithm,
        sampler = _sampler
        )



