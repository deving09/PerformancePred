Fine-Tuned V2-Top Discriminator: 'accuracy': 0.6646245121955872, 'auc': 0.6534550441553381
Fine-Tuned V2-Matched Discriminator: 'accuracy': 0.6720355749130249, 'auc': 0.6799535975581634
Fine-Tuned V2-Threshold Discriminator: 'accuracy': 0.6544466614723206, 'auc': 0.6487832736508776

Linear model V2-Top Discriminator: 'accuracy': 0.6016798615455627, 'auc': 0.5390985097877719
Linear model V2-Matched Discriminator: 'accuracy': 0.6045454740524292, 'auc': 0.5432028183358468
Linear model V2-Threshold Discriminator: 'accuracy': 0.5991106629371643, 'auc': 0.544362372721784



python evaluate_discriminator.py --batch 32 --pretrained --net resnet18  --split 0.5 --cuda 0 --layer_probe penultimate --disc-wt models/imagenet_imagenetv2-top-fixed_disc_standard_resnet18_penultimate_True_linear_False_full_opt_764c12bbaa739a48.pth --source imagenet --target imagenetv2-top-fixed  

pip install -r requirements.txt

mkdir pd_data 
mkdir results 
mkdir pp_logging 
mkdir data 
mkdir experiments
cp _datasets.json datasets.json 
python utils/db.py 
