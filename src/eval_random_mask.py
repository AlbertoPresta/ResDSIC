import wandb
import random
import sys
import matplotlib.pyplot as plt
from compress.utils.parser import parse_args_eval
import torch
from   compress.training.image.step import  compress_with_ac
from compress.models import get_model
import numpy as np
import seaborn as sns
import os

palette = sns.color_palette("tab10")


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False



def plot_list_rd(bpp_list, psnr_list):


    legenda = []
    colores = []
    diffsymbols = []
    markersizes = []
    for i in range(len(bpp_list)):
        colores.append(palette[i])
        diffsymbols.append(["*"]*40)
        markersizes.append([5]*40)
        if i == 0:
            legenda.append("std")
        elif i == 1:
            legenda.append("scalable")
        else:
            st = "rand_" + str(i - 2) 
            legenda.append(st)
    
    
    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0
    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    for i in range(len(bpp_list)):
        bpp = bpp_list[i]
        psnr = psnr_list[i]
        colore = colores[i]
        leg = legenda[i]

        bpp = torch.tensor(bpp).cpu()
        psnr = torch.tensor(psnr).cpu()
    
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=7)
        #for x, y, marker, markersize_t in zip(bpp, psnr, symbols, markersize):
        plt.plot(bpp, psnr, marker="o", markersize=7, color =  colore)

        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]

    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)
    plt.grid(True)
    wandb.log({"model":0, "model/rate distorsion trade-off": wandb.Image(plt)})
    plt.close()  
    print("FINITO")





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
    wandb.init( config= new_args, project="EVAL_RANDOM_MASK", entity="albipresta")   
    if new_args.seed is not None:
        torch.manual_seed(new_args.seed)
        random.seed(new_args.seed)




    test_pth = "/scratch/dataset/kodak"
    filelist = [os.path.join(test_pth,f) for f in os.listdir(test_pth)]


    net = get_model(new_args,device, lmbda_list)
    net.load_state_dict(checkpoint["state_dict"],strict = True) #dddfff
    net.update() 

    if new_args.model in ("progressive","progressive_enc","progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False



    print("entro qua!!!!!")
    #list_pr = [2,10] #list(np.arange(0,10.2,0.2)) 

    bpp_total = []
    psnr_total =  []
    for i in range(5):
        print("************  ",i," ************************************")
        if i == 0:
            mask_pol = "point-based-std"
            list_pr = list(np.arange(0,10.2,0.2)) 
        elif i == 1:
            mask_pol = "scalable_res"
            list_pr = list(np.arange(0,10.2,0.2))
        else: 
            mask_pol = "random"
            list_pr = list(np.arange(0,10.2,0.2)) #sss

        bpp, psnr = compress_with_ac(net,  #ddds
                                 filelist, 
                                 device, 
                                 epoch = -10, 
                                 pr_list = list_pr,   
                                 writing = None, #"/scratch/ScalableResults", 
                                 mask_pol=mask_pol, 
                                 progressive=progressive)
        print(i,": ",bpp," ",psnr)
        bpp_total.append(bpp)
        psnr_total.append(psnr) 
    

    plot_list_rd(bpp_total, psnr_total)
    print("finito tutto")


if __name__ == "__main__":  
    main(sys.argv[1:])