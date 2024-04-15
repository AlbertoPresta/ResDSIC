
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
from   compress.training.image.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.image.loss import ScalableRateDistortionLoss, ScalableMutualRateDistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import get_model, initialize_model_from_pretrained
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
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)


def main(argv):
    args = parse_args(argv)
    print(args)


    dict_base_model = {"q2":"/scratch/base_devil/weights/q2/model.pth",
                       "q5":"/scratch/base_devil/weights/q5/model.pth"}

    
    if  args.model == "restcm": 
        wandb.init( config= args, project="ResDSIC-tcm", entity="albipresta") 
    else:
        wandb.init( config= args, project="ResDSIC-mask", entity="albipresta")  #dddd  
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

    if args.lmbda_list == []:
        lmbda_list = None
    else:
        lmbda_list = args.lmbda_list

    net = get_model(args,device, lmbda_list)
    if args.model in ("restcm","progressive","progressive_enc","progressive_res","progressive_maks","progressive_res","channel"):
        progressive = True
    else:
        progressive = False


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    if "tcm" in args.model:
        print("multistep")
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [45,48], gamma=0.1, last_epoch=-1) 
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    

    if args.mutual:
        criterion = ScalableMutualRateDistortionLoss(lmbda_list=args.lmbda_list)
    else:
        criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)
    

    if args.checkpoint != "none" and args.continue_training:
        print("entro qua e continuo il training")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        new_args = checkpoint["args"]
        net = get_model(new_args,device, lmbda_list)
        net.load_state_dict(checkpoint["state_dict"],strict = True) 
        net.update() 
        last_epoch = checkpoint["epoch"]


    elif  args.checkpoint != "none" and args.continue_training is False:  # load from previous checkpoint

        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k in list(checkpoint["state_dict"]):
            if "entropy" in k or "gaussian" in k:
                print(k)
        #net.update()
        net.load_state_dict(checkpoint["state_dict"], strict=True)
    elif args.checkpoint_base != "none":

        checkpoin_base_model = dict_base_model[args.checkpoint_base]
        base_checkpoint = torch.load(checkpoin_base_model,map_location=device)
        new_check = initialize_model_from_pretrained(base_checkpoint, args.multiple_hyperprior)
        net.load_state_dict(new_check,strict = False)
        net.update() 
        if args.freeze_base:
            net.freeze_base_net(args.multiple_hyperprior)      
    




    best_loss = float("inf")
    counter = 0
    epoch_enc = 0



    if args.tester: 
        #net.freezer(total = True)
        for p in net.parameters():
            p.requires_grad = False
        
        net.print_information()



        print("entro qua!!!!!")
        list_pr = [0,0.5,1]
        mask_pol = "scalable_res" 
        bpp, psnr, _= compress_with_ac(net,  filelist, device, epoch = 0, pr_list = list_pr,   writing = None, mask_pol=mask_pol)
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
        pr_list = [0,1] if "three-levels" not in net.mask_policy else [0,1,2]
        loss = valid_epoch(epoch, 
                           valid_dataloader,
                           criterion, 
                           net, pr_list = pr_list, progressive=progressive)
        print("finito il valid della epoca")
        if "tcm" in args.model:
            lr_scheduler.step()
        else:
            lr_scheduler.step(loss)
            print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')
        

        
        
        
        if epoch_enc > 5 and args.mask_policy not in ("learnable-mask-nested"):
            list_pr = [0,0.2,0.5,0.7,1]
            mask_pol = "point-based-std" 
        else:
            list_pr = [0,1,2,3]  if  "learnable-mask" in args.mask_policy  else [0,1] #ddd
            mask_pol = None 

        if "progressive_mask" == args.model and args.mask_policy == "point-based-std":
            list_pr = net.quality_list
            mask_pol = net.mask_policy
        elif "progressive" in args.model and "mask" not in args.model:
            list_pr = [0,1.5,2,7,10]
            mask_pol = "point-based-std"
        
        if "channel" in args.model:
            list_pr = [0,1,3,5,7,10]
            mask_pol = "point-based-std"
        
        if "progressive_enc" in args.model: #ddd
            list_pr = [0,1.5,2.5,5,6.5,7.5,10]
            mask_pol = "point-based-std"    

        if  args.mask_policy =="three-levels-std":
            mask_pol = "point-based-std" 
            list_pr = [0,1.5,2.5,5,6.5,7.5,10]  

        if  args.mask_policy =="three-levels-learnable":
            mask_pol = "three-levels-learnable"
            list_pr = [0,1,2]  


        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       criterion, 
                       net, 
                       pr_list = list_pr ,  
                       mask_pol = mask_pol,
                       progressive=progressive)
        print("finito il test della epoca: ",bpp_t," ",psnr_t)


        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if True: #epoch%5==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr,_ = compress_with_ac(net,  
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

            if "tcm" not in args.model:
                psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
                bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]
            else:
                if args.N == 128:
                    psnr_res["base"] = [30.07, 30.85, 32.59, 34.33, 36.15, 38.07]
                    bpp_res["base"] = [0.155, 0.194, 0.300, 0.443, 0.625, 0.880]
                elif args.N == 96: 
                    psnr_res["base"] = [29.88, 30.62, 32.38, 34.15, 36.10, 37.98]
                    bpp_res["base"] =  [0.158, 0.195, 0.295, 0.439, 0.623, 0.882]  
                else: 
                    psnr_res["base"] = [29.99, 30.57, 32.42, 34.18, 36.01, 37.96]
                    bpp_res["base"] =  [0.161, 0.198, 0.306, 0.455, 0.628, 0.899]                                   

            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="compression")
            
            bpp_res["our"] = bpp_t
            psnr_res["our"] = psnr_t          
            
            
            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="model")
            epoch_enc += 1



        stringa_lambda = ""
        for lamb in args.lmbda_list:
            stringa_lambda = stringa_lambda  + "_" + str(lamb)


        name_folder = args.code + "_" + stringa_lambda + "_"
        
        
        cartella = os.path.join(args.save_path,name_folder) #dddd


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
                    "aux_optimizer":aux_optimizer.state_dict() if aux_optimizer is not None else "None",
                    "args":args
      
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





if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])
