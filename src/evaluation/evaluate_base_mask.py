import wandb
import random
import sys
import matplotlib.pyplot as plt
from compress.utils.parser import parse_args_eval
import torch
from   compress.training.image.step import  compress_with_ac
from torchvision import transforms
from compress.datasets.utils import  TestKodakDataset
from compress.models import get_model
from compress.models import WACNN
from compress.utils.plot import plot_rate_distorsion, plot_decoded_time
import numpy as np
from compress.result_list import *
import seaborn as sns


palette = sns.color_palette("tab10")


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

from collections import OrderedDict

def replace_keys(checkpoint, multiple_encoder):
    # Creiamo un nuovo OrderedDict con le chiavi modificate all'interno di un ciclo for
    nuovo_ordered_dict = OrderedDict()
    for chiave, valore in checkpoint.items():
        if multiple_encoder:
            if "g_a_enh." in chiave: 
                
                nuova_chiave = chiave.replace("g_a_enh.", "g_a.1.")
                nuovo_ordered_dict[nuova_chiave] = valore
            elif "g_a." in chiave and "g_a.0.1.beta" not in list(checkpoint.keys()): 
                nuova_chiave = chiave.replace("g_a.", "g_a.0.")
                nuovo_ordered_dict[nuova_chiave] = valore  
            else: 
                nuovo_ordered_dict[chiave] = valore   
        else:
            nuovo_ordered_dict[chiave] = valore  
    return nuovo_ordered_dict      





def main(argv):
    args = parse_args_eval(argv)
    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"

    wandb.init( config= args, project="EVAL", entity="albipresta")   

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path

    psnr_res = {}
    bpp_res = {}
    decoded_time = {}

    name_dict = {"res_m4_pret_005_05_memd_frozen":"m4_pret_frozen",
                 "res_m4_005_06_encdec_blocked_kd9":"m4_pret_frozen_kd",
                 "res_m4_005_05_encdec":"m4_memd_005",
                 "res_m4_0025_05_encdec":"m4_memd_0025",
                 "res_m2_md":"m2_md"}
    

    # use base models 

    qs = ["q5","q4"]

    bpp_total = []
    psnr_total = []

    for q in qs:
        pth = "/scratch/base_devil/weights/" + q + "/model.pth"
        checkpoint = torch.load(pth, map_location=device)
        
        net = WACNN()
        net.load_state_dict(checkpoint ,strict = True) #dddfffffffff
        net.update()
        net = net.to(device)
        mask_pol ="point-based-std"

        lista = [10,4,2,1.5,1,0.5,0.1] if q == "q5" else [10]
        bpp, psnr,dec_time = compress_with_ac(net,  
                                    filelist, 
                                    device, 
                                    epoch = -10, 
                                    pr_list = lista,   
                                    writing = None,  
                                    mask_pol=mask_pol, 
                                    progressive=True,
                                    base = True)
        
        bpp_total = bpp_total + bpp
        psnr_total = psnr_total + psnr
    
    bpp_total = bpp_total[::-1]
    psnr_total = psnr_total[::-1]

    print("INIZIANO LA NOSTRA VERSIONE************************")
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"
    total_path = args.path + args.checkpoint[0] + args.model
    print("Loading", total_path)
    checkpoint = torch.load(total_path, map_location=device)
    new_args = checkpoint["args"]

    if "multiple_encoder" not in new_args:
        new_args.multiple_encoder = False
    else: 
        print("l'args ha il multiple encoder!")



    if "multiple_hyperprior" not in new_args:
        new_args.multiple_hyperprior = False
    else: 
        print("l'args ha il multiple_hyperprior!")


    lmbda_list = new_args.lmbda_list
    wandb.init( config= new_args, project="EVAL", entity="albipresta")   

    if new_args.seed is not None:
        torch.manual_seed(new_args.seed)
        random.seed(new_args.seed)
    

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path




    net = get_model(new_args,device, lmbda_list)

    #net = WACNN()
    
    
    checkpoint_model = checkpoint["state_dict"]# replace_keys(checkpoint["state_dict"])
    net.load_state_dict(checkpoint_model ,strict = True) #dddfffffffff
    #net.load_state_dict(checkpoint,strict = True)
    net.update() 

    
    if new_args.model in ("progressive","progressive_enc","progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False


    for p in net.parameters():
        p.requires_grad = False
        
    net.print_information()



    print("entro qua!!!!!")
    list_pr_1 = list(np.arange(0,5.05,0.05))  #dddd
    list_pr_2  = list(np.arange(5,10.5,0.5))


    

    list_pr = list_pr_1 + list_pr_2


    

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


    psnr_res["base"] =  psnr_total
    bpp_res["base"] =  bpp_total
    bpp_res["our"] = bpp
    psnr_res["our"] = psnr


    plot_rate_distorsion(bpp_res, psnr_res,0, eest="compression")


if __name__ == "__main__":  
    main(sys.argv[1:])
