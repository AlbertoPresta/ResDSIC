import sys
import torch
from compress.utils.parser import parse_args

import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
#rc('font', family='Times New Roman')

from compress.datasets.utils import  TestKodakDataset
from compress.zoo import models
import wandb
from torchvision import transforms
from compress.training.step import compress_with_ac

from compress.utils.parser import parse_args
from compress.utils.plot import plot_rate_distorsion
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False




def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net



def main(argv):

    psnr_res = {}  
    bpp_res = {}  


    psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
    bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]


    args = parse_args(argv)
    print(args,"cc")



    device = "cuda" if  torch.cuda.is_available() else "cpu"

    test_transforms = transforms.Compose([transforms.ToTensor()])

    
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform = test_transforms)
    filelist = test_dataset.image_path





    #psnr_res["base"].append(35.91)
    #bpp_res["base"].append(0.649)  
    #legenda["base"]["symbols"] = ["o"]*6 #.append('o')
    #legenda["base"]["markersize"] = [7]*6 #.append(7)



    pth_completed = "/scratch/ResDSIC/models/zero__2_shared_two-levels_320_192_0.0035_0.05_False/0127__lambda__0.05__epoch__30_.pth.tar"
    
    checkpoint = torch.load(pth_completed, map_location=device)

    save_args  = checkpoint["args"]
    net = models[save_args.model](N = save_args.N,
                             M = save_args.M,
                              mask_policy = "point-based-std",
                              lmbda_list = save_args.lmbda_list,
                              independent_lrp = save_args.ind_lrp, 

                              
                             )
        
    net = net.to(device)
    
    net.update() 
    del checkpoint["state_dict"]["entropy_bottleneck._quantized_cdf"]
    del checkpoint["state_dict"]["entropy_bottleneck_prog._quantized_cdf"]
    net.load_state_dict(checkpoint["state_dict"], strict = False)
    net.update() 

    print("ho finito il caricamento ")


    quality_lev = [0,0.5,0.4,0.3,0.2,0.1,1]



    bpp, psnr = compress_with_ac(net,  filelist, device, epoch = -1, pr_list = quality_lev,   writing = None)

    bpp_res["our"] = bpp 
    psnr_res["our"] = psnr

    print(len(bpp_res["our"]),"   ",len(psnr_res["our"]))
    epoch_enc = 0
    plot_rate_distorsion(bpp_res, psnr_res,epoch_enc)
    print("finito il plot")









if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    wandb.init(project="RD-TRADEOFF-ReS", entity="albipresta") 
    main(sys.argv[1:])



