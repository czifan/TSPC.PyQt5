from torch.utils.data import Dataset
from PIL import Image 
import torch 
import numpy as np 
import os 
from torchvision.transforms import transforms
from glob import glob
import numpy as np
from datetime import datetime
import pandas as pd 
import json 
from datetime import datetime
import SimpleITK as sitk
import torch.nn.functional as F

ignored_files = ['2072855_v_54.nii.gz',
 '2112229_v_58.nii.gz',
 '2013097_v_58.nii.gz',
 '2046651_v_58.nii.gz',
 '2113230_QI2TVJFE_128.nii.gz',
 '2041058_v_57.nii.gz',
 '2029174_v_57.nii.gz',
 '2081643_v_54.nii.gz',
 '2047125_v_59.nii.gz',
 '2110342_3PROMHGK_146.nii.gz',
 '2103888_YJQ4QHDQ_124.nii.gz',
 '2071364_XMC0OFSY_86.nii.gz',
 '2116321_ARHRQPKQ_136.nii.gz',
 '2013775_v_None.nii.gz',
 '2056249_v_61.nii.gz',
 '1025794_v_61.nii.gz',
 '823721_v_58.nii.gz',
 '2105833_40024WZU_135.nii.gz',
 '2116746_YMKJZK53_125.nii.gz',
 '2004259_v_64.nii.gz',
 '2021543_v_51.nii.gz',
 '2057385_v_60.nii.gz',
 '2099042_v_56.nii.gz',
 '896452_KRT1OX5W_74.nii.gz',
 '2062335_v_53.nii.gz',
 '2122966_RW05CX1N_113.nii.gz',
 '2062615_v_57.nii.gz',
 '2078041_v_56.nii.gz',
 '2011054_v_58.nii.gz',
 '2005669_v_57.nii.gz',
 '2073726_QJPNXBJM_262.nii.gz',
 '938701_v_57.nii.gz',
 '2040916_v_55.nii.gz',
 '2009967_v_53.nii.gz',
 '945604_v_58.nii.gz',
 '2129184_1ULK4PV5_121.nii.gz',
 '2105692_P0A5NUHH_118.nii.gz',
 '2013775_v_50.nii.gz',
 '2011932_v_56.nii.gz',
 '2127735_NSGCZRID_128.nii.gz',
 '2079870_v_63.nii.gz',
 '1010111_v_56.nii.gz',
 '2079345_JPQLQUBI_92.nii.gz',
 '2133433_KNPCJP2E_113.nii.gz',
 '2073728_v_56.nii.gz',
 '2037104_v_59.nii.gz',
 '2138765_AWGBD5R1_118.nii.gz',
 '2048334_v_63.nii.gz',
 '2069242_v_55.nii.gz',
 '2147387_v_65.nii.gz',
 '2131127_NVUDS1K1_85.nii.gz',
 '2035451_IIA1NBF1_85.nii.gz',
 '2047945_v_58.nii.gz',
 '2081215_v_59.nii.gz',
 '2052643_v_63.nii.gz',
 '2085235_v_60.nii.gz',
 '2125927_CZVLN4RD_74.nii.gz',
 '2059365_v_59.nii.gz',
 '2133700_v_61.nii.gz',
 '2149612_v_61.nii.gz',
 '2006188_EIVCH5AS_78.nii.gz',
 '2031980_CMTS5DJL_91.nii.gz',
 '2080519_v_59.nii.gz']

def random_shift(x, H_random_ratio=0.2, W_random_ratio=0.2):
    # x: (N, 1, 96, 96)
    N, C, H, W = x.shape
    H_shift = np.random.randint(0, int(H_random_ratio*H))
    W_shift = np.random.randint(0, int(W_random_ratio*W))
    if H_shift > 0:
        if np.random.uniform(0, 1) < 0.5:
            x = x[:, :, :-H_shift, :]
            x = torch.cat([torch.zeros(N, C, H_shift, W).float().to(x.device), x], dim=2)
        else:
            x = x[:, :, H_shift:, :]
            x = torch.cat([x, torch.zeros(N, C, H_shift, W).float().to(x.device)], dim=2)
    if W_shift > 0:
        if np.random.uniform(0, 1) < 0.5:
            x = x[:, :, :, :-W_shift]
            x = torch.cat([torch.zeros(N, C, H, W_shift).float().to(x.device), x], dim=3)
        else:
            x = x[:, :, :, W_shift:]
            x = torch.cat([x, torch.zeros(N, C, H, W_shift).float().to(x.device)], dim=3)
    return x

class L3LocDataset(Dataset):
    def __init__(self, data_dir, split, split_file, printer=print,
                 window_level=250, window_width=1000, target_size=96, sigma=1):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.files = [file.strip()+".nii.gz" for file in open(split_file, "r").readlines()]
        self.files = [file for file in self.files if file not in ignored_files]
        self.printer = printer
        self.window_level = window_level
        self.window_width = window_width
        self.hu_lower = self.window_level - self.window_width/2
        self.hu_higher = self.window_level + self.window_width/2
        self.target_size = target_size
        self.sigma = sigma
        
        self.printer(f"[{self.split}] Loaded {self.__len__()} Samples")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_dir, file)))
        
        image = image.clip(self.hu_lower, self.hu_higher)
        image = ((image - self.hu_lower) / (self.hu_higher - self.hu_lower) * 255).astype(np.uint8)
        tmp = image[image.shape[0]//2]
        ind = np.where(tmp)
        y1, y2 = min(ind[0]), max(ind[0])
        x1, x2 = min(ind[1]), max(ind[1])
        image = image[:, y1:y2, x1:x2]
        image = torch.Tensor(image).unsqueeze(dim=1) # (N, 1, H, W)
        X = F.interpolate(image, size=(self.target_size, self.target_size), mode="bilinear") # (N, 1, 96, 96)
        
        slice_id = int(file.split("_")[-1].split(".")[0])
        N = X.shape[0]
        y = torch.zeros(N).float()
        for n in range(N):
            y[n] = 1/((2*np.pi)**0.5*self.sigma)*np.exp(-(n-slice_id)**2/(2*self.sigma**2))
        y /= y.max()
        
        if self.split in ["train",]:
            # random crop
            min_id = np.random.randint(0, slice_id)
            max_id = np.random.randint(slice_id+1, X.shape[0])
            X = X[min_id: max_id]
            y = y[min_id: max_id]
            slice_id = slice_id - min_id
            
            # random shift
            X = random_shift(X)
        return {
            "X": X,
            "slice_id": slice_id,
            "y": y,
        }
    
class L3LocDatasetV1(L3LocDataset):
    def __getitem__(self, index):
        file = self.files[index]
        
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_dir, file)))
        N = image.shape[0]
        image = image[:, :, image.shape[2]//2]
        image = image.clip(self.hu_lower, self.hu_higher)
        image = ((image - self.hu_lower) / (self.hu_higher - self.hu_lower) * 255).astype(np.uint8) # (H, W)
        image = torch.Tensor(image).unsqueeze(dim=0).unsqueeze(dim=0) # (1, 1, H, W)
        X = F.interpolate(image, size=(self.target_size, self.target_size), mode="bilinear")[0] # (1, 96, 96)
        
        slice_id = int(file.split("_")[-1].split(".")[0])
        y = torch.zeros(N).float()
        for n in range(N):
            y[n] = 1/((2*np.pi)**0.5*self.sigma)*np.exp(-(n-slice_id)**2/(2*self.sigma**2))
        y /= y.max()
        
        return {
            "X": X, # (C, H, W)
            "slice_id": slice_id,
            "y": y, # (N,)
        }

        
if __name__ == "__main__":
    ds = L3LocDataset("Data/L3LocData", split="train", split_file="Data/train_L3.txt",
                      printer=print, sigma=2)
    for d in ds:
        X = d["X"]
        y = d["y"]
        slice_id = d["slice_id"]
        print(X.shape)
        print(slice_id, y)
        break
        