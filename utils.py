import csv
import os
import shutil
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, sip
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from xlsxwriter.workbook import Workbook
import SimpleITK as sitk 
from time import sleep
import qtawesome
import cv2
import shutil
from copy import deepcopy
import subprocess
import xlwt
import logging
from PIL import Image, ImageQt
import torch 
import torch.nn as nn
from torchvision.models import resnet18

label_to_id = {
    'BACKGROUND': 0,
    'MPSI': 1,
    'MPSO': 2,
    'MVEN': 3,
    'SAT': 4,
    'VAT': 5,
}

cmap = np.array(
    [
        (0, 0, 0),
        (255, 255, 0),
        (0, 205, 0),
        (72, 118, 255),
        (0, 0, 139),
        (255, 0, 0),
    ],
    dtype=np.uint8,
)

def read_dcm(dcm_dir):
    reader = sitk.ImageSeriesReader()
    img_name = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(img_name)
    image = reader.Execute()
    return image

class MyThread(QThread):
    signalForText = pyqtSignal(str)

    def __init__(self, data=None, parent=None):
        super(MyThread, self).__init__(parent)
        self.data = data

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):
        log = os.popen(self.data)
        print(log.read())

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

class L3LocModel(nn.Module):
    def __init__(self, N_neighbor=2, d_model=512):
        super().__init__()
        
        backbone = resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(1, backbone.conv1.out_channels,
                                  kernel_size=backbone.conv1.kernel_size,
                                  stride=backbone.conv1.stride,
                                  bias=backbone.conv1.bias)
        self.pool = None
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(d_model*(1+2*N_neighbor), d_model),
            nn.Linear(d_model, 1)
        )
        self.N_neighbor = N_neighbor

    def forward(self, x, N_lst=None):
        # x: (B, N, 3, 96, 96)
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        cnn_feat = self.cnn(x)
        if self.pool:
            cnn_feat = self.pool(cnn_feat)
        cnn_feat = cnn_feat.view(B, N, 1, -1) # (B, N, 1, 512)
        
        feat = []
        for n in range(-self.N_neighbor, self.N_neighbor+1): # (-2, -1, 0, 1, 2)
            if n <= 0: 
                tmp = cnn_feat[:, abs(n):, ...]
                tmp = torch.cat([tmp, torch.zeros(B, abs(n), *cnn_feat.shape[-2:]).float().to(cnn_feat.device)], dim=1)
            else: 
                tmp = cnn_feat[:, :-n, ...]
                tmp = torch.cat([torch.zeros(B, n, *cnn_feat.shape[-2:]).float().to(cnn_feat.device), tmp], dim=1)
            feat.append(tmp)
        feat = torch.cat(feat, dim=2) # (B, N, 1+2*N_neighbor, 512)
        feat = feat.view(B, N, -1)
        pred = self.classifier(feat).squeeze(dim=2)
        return pred