import wandb
import random
import sys
from compress.utils.functions import  create_savepath
from compress.utils.parser import parse_args_mask
from compress.utils.plot import plot_rate_distorsion
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   compress.training.image.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.image.loss import  DistortionLoss, ScalableRateDistortionLoss, MaskRateDistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import get_model
import os

import seaborn as sns

palette = sns.color_palette("tab10")
import matplotlib.pyplot as plt

def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression"):


    legenda = {}
    legenda["base"] = {}
    legenda["our"] = {}





    legenda["base"]["colore"] = [palette[0],'-']
    legenda["base"]["legends"] = "reference"
    legenda["base"]["symbols"] = ["*"]*6
    legenda["base"]["markersize"] = [5]*6

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "proposed"
    legenda["our"]["symbols"] = ["*"]*len(psnr_res["our"])
    legenda["our"]["markersize"] = [5]*len(psnr_res["our"])



    
    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(bpp_res.keys()) #[base our]



    list_names = list(bpp_res.keys())

    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        symbols = legenda[type_name]["symbols"]
        markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]


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

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    if eest == "model":
        wandb.log({"model":epoch,
              "model/rate distorsion trade-off": wandb.Image(plt)})
    else:  
        wandb.log({"compression":epoch,
              "compression/rate distorsion trade-off": wandb.Image(plt)})       
    plt.close()  
    print("FINITO")


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])






class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }





    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    if args.only_mask is False:
        aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    else:
        aux_optimizer = None

    return optimizer, aux_optimizer




def save_checkpoint(state, is_best, last_pth,very_best):
    if is_best:
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)


def main(argv):
    args = parse_args_mask(argv)
    print(args)


    wandb.init( config= args, project="ResDSIC-mask", entity="albipresta")   
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=train_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path

    device = "cuda" 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    pret_args = checkpoint["args"]
    lmbda_list = pret_args.lmbda_list if "quantile" in args.mask_policy else args.lmbda_list
    if "multiple_encoder" not in pret_args:
        pret_args.multiple_encoder = False

    pret_args.mask_policy = args.mask_policy # impongo una mask policy diversa!!!

    if args.pretrained is False: 
        pret_args.support_progressive_slices = 4
    net = get_model(pret_args,device, lmbda_list)

    if args.pretrained:
        net.load_state_dict(checkpoint["state_dict"],strict = False) #deve essere falso altrimenti mi da errore
    net.update() 
    if pret_args.model in ("progressive","progressive_enc","progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)


    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    criterion = MaskRateDistortionLoss(lmbda=args.lmbda_list[1])


    net.update()  


    list_quality = args.list_quality  #if "quantile" in args.mask_policy else None
    best_loss = float("inf")
    counter = 0
    epoch_enc = 0


    if args.only_mask:
        net.only_mask()



    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        mask_pol = args.mask_policy
        start = time.time()
        num_tainable = net.print_information()
        
        if num_tainable > 0:
            counter = train_one_epoch( net, 
                                      criterion, 
                                      train_dataloader, 
                                      optimizer, 
                                      aux_optimizer, 
                                      epoch, 
                                      counter,
                                      list_quality = list_quality,
                                      sampling_training = args.sampling_training)
            
        print("finito il train della epoca")

        loss = valid_epoch(epoch, 
                           valid_dataloader,
                           criterion, 
                           net, 
                           pr_list = [0] + net.quality_list,
                           mask_pol= args.mask_policy,
                        progressive=progressive)
        print("finito il valid della epoca")

        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')
        

        
        mask_pol = args.mask_policy

        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       criterion, 
                       net, 
                       pr_list =[0] + net.quality_list + [2], #list_quality + [10], #sss
                       mask_pol = mask_pol,
                       progressive=progressive)
        print("finito il test della epoca: ",bpp_t," ",psnr_t)

        is_best =  loss < best_loss
        best_loss =  min(loss, best_loss)
        

        if epoch%5==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr,_= compress_with_ac(net,  
                                        filelist, 
                                        device,
                                        epoch = epoch_enc, 
                                        pr_list =[0.0] + net.quality_list + [2],  
                                        mask_pol = mask_pol,
                                        writing = None,
                                        progressive = progressive)
            psnr_res = {}
            bpp_res = {}

            bpp_res["our"] = bpp
            psnr_res["our"] = psnr

            psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
            bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="compression")
            
            
            bpp_res["our"] = bpp_t
            psnr_res["our"] = psnr_t          
            
            
            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="model")
            
            epoch_enc += 1



        if pret_args.checkpoint != "none":
            check = "pret"
        else:
            check = "zero"

        #stringa_lambda = ""
        #for lamb in new_args.lmbda_list:
            #stringa_lambda = stringa_lambda  + "_" + str(lamb)


        name_folder = check + "_" + args.checkpoint.split("/")[-2] + "_" + args.mask_policy + \
                     str(args.only_mask) + "_" + str(args.pretrained)
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.")  #ddddfffffrirririririr
        else:
            print(f"La cartella '{cartella}' esiste già.")


        last_pth, very_best =  create_savepath( cartella)



        #if is_best is True or epoch%10==0 or epoch > 98: #args.save:
        """
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer":aux_optimizer if aux_optimizer is not None else "None",
                    "args":args
      
                },
                is_best,
                last_pth,
                very_best
                )
        """
        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr']
        #"train/beta": annealing_strategy_gaussian.bet
        }

        wandb.log(log_dict)



        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)       





if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])
