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






def create_savepath(args,epoch,base_path):
    now = datetime.now()
    date_time = now.strftime("%m%d")
    c = join(date_time,"_lambda_",str(args.lmbda_starter),"_epoch_",str(epoch)).replace("/","_")

    
    c_best = join(c,"best").replace("/","_")
    c = join(c,".pth.tar").replace("/","_")
    c_best = join(c_best,".pth.tar").replace("/","_")
    
    
    
    savepath = join(base_path,c)
    savepath_best = join(base_path,c_best)
    
    print("savepath: ",savepath)
    print("savepath best: ",savepath_best)
    very_best  = join(base_path,"_very_best.pth.tar")
    return savepath, savepath_best, very_best








def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


