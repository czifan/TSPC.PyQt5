from turtle import forward
import scipy
import scipy.stats
import numpy as np
import logging 
import torch 
import random 
import torch.nn as nn 
import math
from sklearn import metrics

def print_args(args, printer):
    for arg in vars(args):
        printer(format(arg, '<20')+'\t'+format(str(getattr(args, arg)), '<')) 

def build_logging(filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

def setup_seed(seed):
    #torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def topK_accuracy(y_pred, y_true, K=1):
    # y_pred: (B,)
    # y_true: (B,)
    dis = np.abs(y_pred - y_true)
    acc = 1.0 * sum(dis <= K) / len(dis)
    return acc

class GaussianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, eps=1e-6):
        # y_pred: (B, N)
        # y_true: (B, N)
        y_pred = torch.softmax(y_pred, dim=1)
        pos_loss = -torch.mean(torch.sum(y_true*torch.log(y_pred+eps), dim=1))
        neg_loss = -torch.mean(torch.sum((1.0-y_true)*torch.log(1-y_pred+eps), dim=1))
        return pos_loss + neg_loss
    
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, y_true):
        # y_pred: (B, N)
        # y_true: (B, N)
        y_true = torch.argmax(y_true, dim=1).long()
        return self.criterion(y_pred, y_true)