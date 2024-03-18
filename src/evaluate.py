import wandb
import random
import sys
import matplotlib.pyplot as plt
from compress.utils.parser import parse_args_eval
import torch
from   compress.training.step import  compress_with_ac
from torchvision import transforms
from compress.datasets.utils import  TestKodakDataset
from compress.models import get_model
from compress.utils.plot import plot_rate_distorsion
import numpy as np

import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#plt.rc('text', usetex=True)
#plt.rc('font', family='Times New Roman')


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False




def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net









def main(argv):
    args = parse_args_eval(argv)
    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"

    print("Loading", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print("zio caro---> ",checkpoint.keys())
    new_args = checkpoint["args"]
    lmbda_list = new_args.lmbda_list

    wandb.init( config= new_args, project="EVAL", entity="albipresta")   
    if new_args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path

    





    net = get_model(new_args,device, lmbda_list)
    #net.update() 
    net.load_state_dict(checkpoint["state_dict"],strict = True) #dddfff
    net.update() 
    if new_args.model in ("progressive","progressive_enc","progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False


    for p in net.parameters():
        p.requires_grad = False
        
    net.print_information()



    print("entro qua!!!!!")
    list_pr = list(np.arange(0,10.2,0.2)) 
    mask_pol ="point-based-std"
    bpp, psnr = compress_with_ac(net,  
                                 filelist, 
                                 device, 
                                 epoch = -10, 
                                 pr_list = list_pr,   
                                 writing = None, #"/scratch/ScalableResults", 
                                 mask_pol=mask_pol, 
                                 progressive=progressive)
    print("*********************************   OVER *********************************************************")
    print(bpp,"  ++++   ",psnr) 


    psnr_res = {}
    bpp_res = {}

    bpp_res["our"] = bpp
    psnr_res["our"] = psnr

    psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
    bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

    
        
        
    bpp_res["tri_planet"] = [0.660875109,
                             0.655721029	
                             ,0.64996677	
                             ,0.64375813,
                        0.636986627	,
                        0.629737006	,
                        0.621955024	,
                        0.613691542	,
                        0.604977078	,
                        0.595689562	,
                        0.585951063	,
                        0.575720893	,
                        0.564975315	,
                        0.553833008	,
                        0.542127821	,
                        0.530036079	,
                        0.517506917	,
                        0.504621718	,
                        0.491258409	,
                        0.477579753	,
                        0.463487413	,
                        0.449042426	,
                        0.434268528	,
                        0.419165717	,
                        0.403811985	,
                        0.388197157	,
                        0.372334798	,
    ][::-1]

    psnr_res["tri_planet"] = [34.80704198,
                            34.76644063,
                            34.71799054,
                            34.66435169,
                            34.60298681,
                            34.53566738,
                            34.46155532,
                            34.3790979,
                            34.28852354,
                            34.19085494,
                            34.08420287,
                            33.96759937,
                            33.84475605,
                            33.71422386,
                            33.57031665,
                            33.42045507,
                            33.26175287,
                            33.09296812,                  
                            32.91922032,
                            32.73944584,
                            32.55061685,
                            32.35301494,
                            32.15103617,
                            31.94190188,
                            31.73074372,
                            31.51412129,
                            31.43688091][::-1]

    plot_rate_distorsion(bpp_res, psnr_res,0, eest="compression")

        




if __name__ == "__main__":  
    main(sys.argv[1:])



