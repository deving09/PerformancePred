import torch
import torch.nn as nn
import numpy as np
import logging

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
#from .helpers import *


def build_disc_labels(samples):
    inputs = [d_samples for d_samples, l in  samples]
    labels = [torch.zeros(d_samples.shape[0], dtype=torch.long) + i for i, (d_samples, l) in enumerate(samples)]

    inputs = torch.cat(tuple(inputs), 0)
    labels = torch.cat(tuple(labels), 0)

    return inputs, labels


def rotate_img(img, rot):
    if rot == 0: # no rotation
        return img
    elif rot == 1: # 90 degree rotation
        return torch.flipud(torch.transpose(img, 1, 2))
    elif rot == 2: # 180 degree rotation
        return torch.fliplr(torch.flipud(img))
    elif rot == 3: # 270 degree rotation
        return torch.transpose(torch.flipud(img), 1, 2)


def build_pretext_labels(samples):

    inputs = []
    labels = []

    for d_samples, l in samples:
        pretext_labels = torch.randint(0, 4, (d_samples.shape[0],))
        
        for temp_idx in range(d_samples.shape[0]):
            d_samples[temp_idx] = rotate_img(d_samples[temp_idx],
                    pretext_labels[temp_idx].item())

        inputs.append(d_samples)
        labels.append(pretext_labels)

    inputs = torch.cat(tuple(inputs), 0)
    labels = torch.cat(tuple(labels), 0)

    return inputs, labels

def build_flatten_labels(samples):
    inputs = []
    labels = []

    for d_samples, l in samples:
        inputs.append(d_samples)
        labels.append(l)

    inputs = torch.cat(tuple(inputs), 0)
    labels = torch.cat(tuple(labels), 0)

    return inputs, labels


def eval_discriminator(discriminator, datasets, eval_params, device=None, build_labels=None):
    evals = {}

    for (eval_type, p) in eval_params:
        if eval_type == "accuracy":
            evals[eval_type] = AccuracyCalculator()
        elif eval_type == "auc":
            evals[eval_type] = ROC_AUC()
        elif eval_type == "brier":
            evals[eval_type] = BrierScore(raw_scores=False) #raw_scores=True)
        elif eval_type == "ece":
            evals[eval_type] = ECELoss(raw_scores=True)
        elif eval_type == "test_avg":
            evals[eval_type] = AvgMaxScore(raw_scores=True)
        elif eval_type == "loss":
            evals[eval_type] = LossAggregator()

    
    for i, samples in tqdm(enumerate(zip(*datasets))):
        if build_labels:
            inputs, targets = build_labels(samples)
        else:
            inputs, targets = build_flatten_labels(samples)

           
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = discriminator(inputs)
            outputs = outputs.cpu()
            for ev in evals.values():
                ev.update(outputs, targets)

    for eval_type, ev in evals.items():
        res = ev.results()
        evals[eval_type] = res.item()

    return evals



def eval_classifier(classifier, dataset, eval_params, 
                    device=None, temperature=1.0,
                    filtering=None, conversion=None,
                    batch_norm=False):


    evals = {}
    raw_scores=True

    for (eval_type, p)  in eval_params:
        if eval_type == "accuracy":
            evals[eval_type] = AccuracyCalculator()
        elif eval_type == "auc":
            evals[eval_type] = ROC_AUC()
        elif eval_type == "brier":
            evals[eval_type] = BrierScore(raw_scores=raw_scores)
        elif eval_type == "ece":
            evals[eval_type] = ECELoss(raw_scores=raw_scores)
        elif eval_type == "test_avg":
            evals[eval_type] = AvgMaxScore(raw_scores=raw_scores)
        elif eval_type == "loss":
            evals[eval_type] = LossAggregator()
        elif eval_type == "mse":
            evals[eval_type] = MSE_Score()
        elif eval_type == "mae":
            evals[eval_type] = MAE_Score()
        elif eval_type == "r2_score":
            evals[eval_type] = R2_Score()
        elif eval_type == "entropy":
            evals[eval_type] = EntropyCalc(raw_scores=raw_scores)

    logging.info(dataset.dataset)
    logging.info(len(dataset))
    logging.info(len(dataset.dataset))

    gen_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ]) 
 
    for i, (inputs, targets) in tqdm(enumerate(dataset)):
        with torch.no_grad():

            if filtering is not None:
                inputs, targets = filtering(inputs, targets)
            
            inputs_numpy = inputs.cpu().numpy()
            inputs_numpy = np.transpose(inputs_numpy, (0,2,3,1)) * 255
            inputs_numpy = inputs_numpy.astype(np.uint8)
            
            bs = inputs_numpy.shape[0]

            for c_idx in range(bs):
                img = inputs_numpy[c_idx]
                inputs[c_idx] = gen_transforms(img)

            inputs = inputs.to(device)
            outputs = classifier(inputs)

            if hasattr(classifier.net, "combine_preds"):
                outputs, targets = classifier.net.combine_preds_eval(outputs, targets, conv=conversion)

            else:
                outputs = outputs / temperature

                outputs = outputs.cpu()
                
                
                if conversion is not None:
                    outputs, targets = conversion(outputs, targets)
            
            for ev in evals.values():
                ev.update(outputs, targets)


    for eval_type, ev in evals.items():
        res = ev.results()
        evals[eval_type] = res.item()


    return evals




class AccuracyCalculator(object):

    def __init__(self, k=1, convert=None):
        self.correct = 0.0
        self.total = 0.0
        self.k = k


    def update(self, output, target):
        """
        Updates accuracy numbers
        """

        with torch.no_grad():

            batch_size = target.shape[0] #.size(0)
            #print(batch_size)

            _, pred = output.topk(self.k, 1, True, True)

            pred = pred.t()

            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct = correct.reshape(-1).float().sum(0)
            self.correct +=  correct
            self.total += batch_size


    def results(self):
        return self.correct / self.total



class EntropyCalc(object):

    def __init__(self, raw_scores=True):
        """
        Calculate the average entropy
        """
        self.raw_scores=raw_scores
        self.total = 0.0
        self.loss = 0.0

    def update(self, output, target):
        
        with torch.no_grad():
            if self.raw_scores:
                output = torch.nn.functional.softmax(output, dim=1)

            self.total += target.shape[0] #.size(0)
            output = torch.clamp(output, min=1e-6, max=1.0)
            entropy = -1 * torch.sum(output * torch.log(output))
            self.loss += entropy

    def results(self):

        ent =  self.loss / self.total
        logging.info("Entropy: %f" % ent.item())
        return ent#.item()



class LossAggregator(object):


    def __init__(self, criterion=nn.functional.cross_entropy):
        self.criterion = criterion
        self.total = 0.0
        self.loss = 0.0



    def update(self, output, target):
        """
        compute losses
        """
        with torch.no_grad():

            batch_size = target.shape[0]
            self.loss += self.criterion(output, target, reduction="sum")
            self.total += batch_size


    def results(self):
        rel = self.loss / self.total
        return rel #.item()


class AvgMaxScore(object):

    def __init__(self, raw_scores=True):
        """
        Calculate Average Max
        """
        self.total = 0.0
        self.max_sum = 0.0
        self.raw_scores = True 


    def update(self, output, target):

        with torch.no_grad():
            if self.raw_scores:
                output = torch.nn.functional.softmax(output, dim=1)

            self.total += target.shape[0]

            cal, preds  = torch.max(output, 1)
            self.max_sum += torch.sum(cal)

    def results(self):
        return self.max_sum / self.total


class BrierScore(object):

    def __init__(self, raw_scores=True):
        """
        Calculate raw scores
        """

        self.total = 0.0
        self.loss = 0.0

        self.raw_scores = raw_scores

    def update(self, output, target):

        with torch.no_grad():
            if self.raw_scores:
                output = torch.nn.functional.softmax(output, dim=1)

            self.total += target.shape[0]

            t_u = target.unsqueeze(-1)
            o_select = torch.gather(output, 1, t_u)
            goal = torch.ones(target.shape)

            self.loss += torch.pow(goal - o_select.T, 2).sum()


    def results(self):
        return self.loss / self.total


class ECELoss(object):


    def __init__(self, n_bins=15, raw_scores=True):

        self.total = 0.0
        self.loss = 0.0

        self.raw_scores = raw_scores
        self.n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.correct_in_bin = torch.zeros((n_bins), dtype=torch.float32)
        self.conf_in_bin = torch.zeros((n_bins), dtype=torch.float32)
        self.total_in_bin = torch.zeros((n_bins), dtype=torch.float32)


    def update(self, output, target):

        if self.raw_scores:
            output = torch.nn.functional.softmax(output, dim=1)

        conf, preds = torch.max(output, 1)
        correct = preds.eq(target)

        self.total += target.shape[0]

        for i,(bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = conf.gt(bin_lower.item()) * conf.le(bin_upper.item())

            self.total_in_bin[i] += in_bin.float().sum()

            if in_bin.float().sum() > 0:
                self.correct_in_bin[i] += correct[in_bin].float().sum()
                self.conf_in_bin[i] += conf[in_bin].sum()


    def results(self):

        ece = 0.0
        for i in range(self.n_bins):
            if self.total_in_bin[i] > 0.0:
                avg_acc = self.correct_in_bin[i] / self.total_in_bin[i]
                avg_conf = self.conf_in_bin[i] / self.total_in_bin[i]
                bin_prob = self.total_in_bin[i] / self.total
                
                ece += torch.abs(avg_acc - avg_conf) * bin_prob

        logging.info("Total in bin")
        logging.info((self.total_in_bin / self.total).numpy())
        logging.info("Conf in bin")
        logging.info((self.conf_in_bin / self.total_in_bin).numpy())
        logging.info("Correct in bin")
        logging.info((self.correct_in_bin / self.total_in_bin).numpy())
        return ece
                

class ROC_AUC(object):

    def __init__(self, binary=True):
        self.y_true = []
        self.y_pred = []
        self.binary = binary


    def update(self, output, target, disc_score=None):

        if self.binary or disc_score is not None:
            conf, preds = torch.max(output, 1)

            correct = preds.eq(target)
            correct = correct.float()

            self.y_true.append(correct)

            if disc_score is not None:
                self.y_pred.append(1 - disc_score)
            else:
                self.y_pred.append(conf)
        else:
            self.y_true.append(target)
            conf, preds = torch.max(output, 1)
            self.y_pred.append(conf)
    
    def results(self):

        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            raise RuntimeError("This module requires sklearn to be installed")


        y_true = torch.cat(self.y_true, 0)
        y_pred = torch.cat(self.y_pred, 0)

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        return roc_auc_score(y_true, y_pred)



class MSE_Score(object):
    
    def __init__(self):
        self.total = 0.0
        self.loss  = 0.0

    def update(self, output, target):

        with torch.no_grad():

            self.total += output.shape[0]
            self.loss += torch.pow(target-output, 2).sum()
    
    def results(self):
        return self.loss/self.total


class WorstError(object):


    def __init__(self):
        self.max_error = 0.0
        #self.loss  = 0.0

    def update(self, output, target):

        with torch.no_grad():

            max_e, _ = torch.max(torch.abs(target-output), 0)
            max_e = max_e.item()
            self.max_error = max([max_e, self.max_error])


    def results(self):
        return self.max_error



class MAE_Score(object):


    def __init__(self):
        self.total = 0.0
        self.loss  = 0.0

    def update(self, output, target):

        with torch.no_grad():

            self.total += output.shape[0]
            self.loss += torch.abs(target-output).sum()

    def results(self):
        return self.loss/self.total


class R2_Score(object):


    def __init__(self):
        self.y_true = []
        self.y_pred = []


    def update(self, output, target):
        
        self.y_true.append(target)
        self.y_pred.append(output)
    
    def results(self):

        try:
            from sklearn.metrics import r2_score
        except ImportError:
            raise RuntimeError("This module requires sklearn to be installed")

        logging.info(self.y_true)
        logging.info(self.y_pred)
        y_true = torch.cat(self.y_true, 0)
        y_pred = torch.cat(self.y_pred, 0)

        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

        return r2_score(y_true, y_pred)







