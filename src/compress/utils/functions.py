import argparse
from compress.zoo import models
from datetime import datetime
from os.path import join   
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math 
import torch

def read_image(filepath,):
    #assert filepath.is_file()
    img = Image.open(filepath)   
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count






def create_savepath(base_path):
    very_best  = join(base_path,"_very_best.pth.tar")
    last = join(base_path,"_last.pth.tar")
    return very_best, last



import numpy as np
def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)




def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


