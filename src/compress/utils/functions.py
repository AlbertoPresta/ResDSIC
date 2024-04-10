
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
    return last, very_best








def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


