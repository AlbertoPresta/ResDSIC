
import wandb
import random
import sys
from compress.utils.functions import  create_savepath
from compress.utils.parser import parse_args
from compress.utils.plot import plot_rate_distorsion
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   compress.training.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.loss import ScalableRateDistortionLoss, RateDistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import get_model
#from compress.zoo import models
import os

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
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer




def save_checkpoint(state, is_best, last_pth,very_best):


    if is_best:
        print("ohhuuuuuuuuuuuuuu veramente il best-------------Z ",very_best)
        torch.save(state, very_best)
        #torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)


def main(argv):
    args = parse_args(argv)
    print(args)

    

    wandb.init( config= args, project="ResDSIC-dsic", entity="albipresta")   
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    print("sono qua arrivato all'inizio")
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

    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"

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

    if args.lmbda_list == []:
        lmbda_list = None
    else:
        lmbda_list = args.lmbda_list

    net = get_model(args,device, lmbda_list)
    if args.model in ("progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False




    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)


    last_epoch = 0
    if args.checkpoint != "none":  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = 0 # checkpoint["epoch"] + 1
        #print("madonna troai")
        for k in list(checkpoint["state_dict"]):
            if "entropy" in k or "gaussian" in k:
                print(k)
        #net.update()
        net.load_state_dict(checkpoint["state_dict"], strict=True)
    elif args.checkpoint_base != "none":
        print("riparto da un modello base------")
        print("Loading", args.checkpoint_base)
        checkpoint = torch.load(args.checkpoint_base, map_location=device)


        if args.multiple_decoder:
            for keys in list(checkpoint.keys()):
                if "g_s" in keys:
                    nuova_chave = "g_s.0"  +keys[3:]
                    checkpoint[nuova_chave] = checkpoint[keys]



        #for keys in list(checkpoint.keys()):
        #    if "g_s" in keys:
        #        print(keys)
        last_epoch = 0 # checkpoint["epoch"] + 1
        #net.update()



        #checkpoint['gaussian_conditional_prog._quantized_cdf'] = net.state_dict()['gaussian_conditional_prog._quantized_cdf']
        net.load_state_dict(checkpoint,strict = False)
        print("ho fatto il salvataggio!!!")#dddd
        net.update()       
        
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)


    if args.checkpoint != "none" and args.continue_training:    
        print("conitnuo il training!")
        optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    


    if args.only_progressive:
        print("entro su freezer la base!")
        net.unfreeze_only_progressive()

    best_loss = float("inf")
    counter = 0
    epoch_enc = 0



    if args.tester: 
        #net.freezer(total = True)
        for p in net.parameters():
            p.requires_grad = False
        
        net.print_information()

        #bpp_test, psnr_test = test_epoch(0, 
        #               test_dataloader,
        #               criterion, 
        #               net, 
        #               pr_list = [0],
        #               mask_pol= "two-levels")
        #print("test:  ",bpp_test,"   ",psnr_test)

        print("entro qua!!!!!")
        list_pr = [0,0.5,1]
        mask_pol = "scalable_res" 
        bpp, psnr = compress_with_ac(net,  filelist, device, epoch = 0, pr_list = list_pr,   writing = None, mask_pol=mask_pol)
        print("*********************************   OVER *********************************************************")
        print(bpp,"  ++++   ",psnr) 
        return 0


    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        num_tainable = net.print_information()
        if num_tainable > 0:
            counter = train_one_epoch( net, 
                                      criterion, 
                                      train_dataloader, 
                                      optimizer, 
                                      aux_optimizer, 
                                      epoch, 
                                      counter,
                                      sampling_training = args.sampling_training)
            
        print("finito il train della epoca")
        loss = valid_epoch(epoch, valid_dataloader,criterion, net, pr_list = [0,1],lmbda_list = args.lmbda_list, progressive=progressive)
        print("finito il valid della epoca")

        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')
        

        
        
        
        if epoch_enc > 5 and args.mask_policy not in ("learnable-mask-nested"):
            list_pr = [0,0.2,0.5,0.7,1]
            mask_pol = "point-based-std" 
        else:
            list_pr = [0,1,2,3]  if  "learnable-mask" in args.mask_policy  else [0,1] #ddd
            mask_pol = None 


        if args.mask_policy == "scalable_res":
            list_pr = [0,1,2,3]
            mask_pol = None
        
        if args.mask_policy == "all-one": #dddd#
            mask_pol = "two-levels"
            list_pr = [0,1]
        

        

        if "progressive_mask" == args.model and args.mask_policy == "point-based-std":
            list_pr = net.quality_list
            mask_pol = net.mask_policy
        elif "progressive" in args.model and "mask" not in args.model:
            list_pr = [0,0.15,0.25,0.5,0.65,0.75,1]
            mask_pol = "point-based-std"
        
        if "channel" in args.model:
            list_pr = [0,0.15,0.25,0.5,0.65,0.75,1]
            mask_pol = "point-based-std"
        
                   


        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       criterion, 
                       net, 
                       pr_list = list_pr ,  
                       mask_pol = mask_pol,
                       progressive=progressive)
        print("finito il test della epoca: ",bpp_t," ",psnr_t)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if epoch%5==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr = compress_with_ac(net,  
                                         filelist, 
                                         device,
                                           epoch = epoch_enc, 
                                           pr_list =list_pr,  
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



        if args.checkpoint != "none":
            check = "pret"
        else:
            check = "zero"

        stringa_lambda = ""
        for lamb in args.lmbda_list:
            stringa_lambda = stringa_lambda  + "_" + str(lamb)


        name_folder = check + "_" + "_multi_" + stringa_lambda + "_" + args.model + "_" +  \
            args.mask_policy +  "_" +   str(args.lrp_prog) + str(args.joiner_policy) + "_" + str(args.sampling_training)
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.")  #ddddfffffrirririririr
        else:
            print(f"La cartella '{cartella}' esiste già.")


        last_pth, very_best =  create_savepath( cartella)



        #if is_best is True or epoch%10==0 or epoch > 98: #args.save:
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer":aux_optimizer if aux_optimizer is not None else "None",
                    "args":args,
                    "epoch":epoch
                },
                is_best,
                last_pth,
                very_best
                )

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
