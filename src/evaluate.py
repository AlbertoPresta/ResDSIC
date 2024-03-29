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
from compress.utils.plot import plot_rate_distorsion, plot_decoded_time
import numpy as np
import seaborn as sns

palette = sns.color_palette("tab10")


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def main(argv):
    args = parse_args_eval(argv)
    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"

    print("Loading", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    new_args = checkpoint["args"]

    if "multiple_encoder" not in new_args:
        new_args.multiple_encoder = False

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
    list_pr = list(np.arange(0,10.1,0.1))  #dddd
    mask_pol ="point-based-std"
    bpp, psnr,dec_time = compress_with_ac(net,  
                                 filelist, 
                                 device, 
                                 epoch = -10, 
                                 pr_list = list_pr,   
                                 writing = None, #"/scratch/ScalableResults", 
                                 mask_pol=mask_pol, 
                                 progressive=progressive)
    print("*********************************   OVER *********************************************************")
    print(bpp,"  ++++   ",psnr," +++++ ",dec_time) 


    psnr_res = {}
    bpp_res = {}
    decoded_time = {}

    decoded_time["our"] = dec_time

    decoded_time["tri_planet_23"] = [2.3024718718869344, 2.426101867109537,
                                      2.55243898762597, 
                                     2.662715111176173, 2.7725952692104108, 2.8762405349148645, 
                                     2.9079313476880393, 2.980673296329303, 9.18038641413053, 
                                     6.93557970225811, 6.211363573869069, 5.869887267549832, 
                                     5.676065286000569, 5.4823808045614335, 5.5328710553822695, 
                                     5.56682376563549]
    bpp_res["our"] = bpp
    psnr_res["our"] = psnr

    psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
    bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

    
    bpp_res["tri_planet_23"] = [0.19599018399677576, 0.2160843743218315, 
                                 0.23966546706211406, 0.2649169921874998, 0.290478022411616, 
                                     0.31530394377531823, 0.6184760199652779, 0.3386248727130074, 
                                0.6184760199652779, 
                                     0.622775607638889, 0.6264399775752316, 0.629531012641059,
                                       0.6320909288194444, 0.6358637734064978, 0.6489329396942514, 
                                       0.6606713189019093]

    
    psnr_res["tri_planet_23"] = [29.966779946152272, 30.245813808118623, 
                                 30.57321667041242, 
                                 30.91983179476929, 31.261272444242884,
                                  31.581467860369358, 35.35794463342364, 31.871966073681232, 
                                      35.35794463342364,
                                        35.387717967053526, 35.41157437546917, 35.43059575484415,
                                        35.445527093923985, 35.46594985204121, 35.526434249041614, 
                                        35.58748106931754]

        
    bpp_res["tri_planet_22"] = [0.660875109,
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

    psnr_res["tri_planet_22"] = [34.80704198,
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

    plot_decoded_time(bpp_res,decoded_time,0,eest="compression")


        




if __name__ == "__main__":  
    main(sys.argv[1:])



