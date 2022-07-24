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