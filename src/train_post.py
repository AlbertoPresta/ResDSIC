

import wandb
import random
import sys
import math
from compress.utils.functions import  create_savepath

from compress.utils.parser import parse_args_post
from compress.utils.plot import plot_rate_distorsion
from torch.nn.functional import mse_loss
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.image.loss import DistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import get_model, models
#from compress.zoo import models
from compress.result_list import tri_planet_22_bpp, tri_planet_22_psnr, tri_planet_23_bpp, tri_planet_23_psnr
import os
from pytorch_msssim import ms_ssim
import torch.nn.functional as F 
from compressai.ops import compute_padding
from compress.utils.functions import AverageMeter, read_image


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


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                    optimizer,  
                    epoch, 
                    counter,
                    mask_pol,
                    list_quality,
                    sampling_training = False,
                    clip_max_norm = 1.0,
                     ):
    
    assert len(list_quality) >= 1 
    if sampling_training is False:
        assert len(list_quality) == 1

    model.train()
    device = next(model.parameters()).device


    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        if sampling_training:
            quality_index =  random.randint(0, len(list_quality) - 1)
            quality = list_quality[quality_index]
            out_net = model(d, quality = quality, mask_pol = mask_pol)
            out_criterion = criterion(out_net, d)
        else:
            
            out_net = model(d, quality = list_quality[0], mask_pol = mask_pol)
            out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].mean().clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())



        wand_dict = {
            "train_batch": counter,
            "train_batch/loss": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
        }
        wandb.log(wand_dict)
        counter += 1

        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].mean().item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
               f"\tAux loss: {0.000:.2f}"
            )
    
    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        }
    wandb.log(log_dict)
    return counter



def valid_epoch(epoch, test_dataloader,criterion, model, pr_list = [0.05], mask_pol = None):

    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter() 
    bpp_loss =AverageMeter() 
    mse_lss = AverageMeter() 
    
    psnr = AverageMeter() 


    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)
            for _,p in enumerate(pr_list):

                out_net = model(d, quality = p, mask_pol = mask_pol,  training = False)

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator
                    
                bpp_loss.update(bpp)

                mse_lss.update(mse_loss(d, out_net["x_hat"]))
                psnr.update(psnr_im) 

                out_criterion = criterion(out_net, d, lmbda = p) #dddddd

                loss.update(out_criterion["loss"].clone().detach())
                           
    log_dict = {
            "valid":epoch, "valid/loss": loss.avg, 
            "valid/bpp":bpp_loss.avg,"valid/mse": mse_lss.avg,
            "valid/psnr":psnr.avg,
            }
    wandb.log(log_dict)
    return loss.avg


def test_epoch(epoch, test_dataloader, model, pr_list, mask_pol = None,post = None):
    model.eval()
    device = next(model.parameters()).device


    bpp_loss =[AverageMeter()  for _ in range(len(pr_list))] 
    psnr = [AverageMeter()  for _ in range(len(pr_list))]


    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            for j,p in enumerate(pr_list):

                post = False  if p > 1 else post 
                out_net = model(d, 
                                quality = p,
                                mask_pol = mask_pol, 
                                training = False)

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator


                psnr[j].update(psnr_im)
                bpp_loss[j].update(bpp)

    for i in range(len(pr_list)):
        if i== 0:
            name = "test_base"
        elif i == len(pr_list) - 1:
            name = "test_complete"
        else:
            c = str(pr_list[i])
            name = "test_quality_" + c 
        
        log_dict = {
            name:epoch,
            name + "/bpp":bpp_loss[i].avg,
            name + "/psnr":psnr[i].avg,
            }

        wandb.log(log_dict)
    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))]

def compress_with_ac(model,  
                     filelist, 
                     device, 
                     epoch, 
                     pr_list = [0.05,0.01], 
                     post = True,
                     mask_pol = None, 
                     cheating = False):

    l = len(pr_list)
    print("ho finito l'update")
    bpp_loss = [AverageMeter() for _ in range(l)]
    psnr =[AverageMeter() for _ in range(l)]
    mssim = [AverageMeter() for _ in range(l)]
    dec_time = [AverageMeter() for _ in range(l)]

    with torch.no_grad():
        for i,d in enumerate(filelist):

            print("image d: ",d)
            x = read_image(d).to(device)
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            for j,p in enumerate(pr_list):
                
                data =  model.compress(x_padded, quality =p, mask_pol = mask_pol )
                start = time.time()
                out_dec = model.decompress(data["strings"], 
                                           data["shape"], 
                                           quality = p, 
                                           mask_pol = mask_pol ) #post = post if p <= 1 else False)
                end = time.time()

                decoded_time = end-start

                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)     

                psnr_im = compute_psnr(x, out_dec["x_hat"])
                ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
                ms_ssim_im = -10*math.log10(1 - ms_ssim_im )
                psnr[j].update(psnr_im)
                mssim[j].update(ms_ssim_im)
                dec_time[j].update(decoded_time)
            
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]

                data_string_scale = data["strings"][0] # questo è una lista
                bpp_scale = sum(len(s[0]) for s in data_string_scale) * 8.0 / num_pixels #ddddddd
                        
                data_string_hype = data["strings"][1]
                bpp_hype = sum(len(s) for s in data_string_hype) * 8.0 / num_pixels

                if cheating is False:
                    bpp = bpp_hype + bpp_scale
                else:
                    bpp = bpp_hype + bpp_scale if j == 0 else bpp_scale

                bpp_loss[j].update(bpp)


    if epoch > -1:
        postit = "post_" if post is True else ""
        for i in range(len(pr_list)):
            if i== 0:
                name = postit + "compress_base"
            elif i == len(pr_list) - 1:
                name = postit + "compress_complete"
            else:
                c = str(pr_list[i])
                name = postit + "compress_quality_" + c
            
            log_dict = {
                name:epoch,
                name + "/bpp":bpp_loss[i].avg,
                name + "/psnr":psnr[i].avg,
                }

            wandb.log(log_dict)



    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))], [dec_time[i].avg for i in range(len(dec_time))]

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
    params_dict = dict(net.named_parameters())
    parameters = {n for n, p in net.named_parameters()if not n.endswith(".quantiles") and p.requires_grad}
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)),lr=args.learning_rate,)
    return optimizer



def save_checkpoint(state, is_best, last_pth,very_best):
    if is_best:
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)




def main(argv):
    args = parse_args_post(argv) 
    wandb.init( config= args, project="ResDSIC-post", entity="albipresta")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    if args.cluster == "hssh":

        openimages_path = "/scratch/dataset/openimages"
        kodak_path = "/scratch/dataset/kodak"
        save_path = "/scratch/ResDSIC/models/"
        path =  "/scratch/ResDSIC/models/" 
    elif args.cluster == "nautilus":

        openimages_path = "/data/openimages"
        kodak_path = "/data/kodak"
        save_path = "/data/models"
        path =  "/data/pretrained/"



    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"
    total_path = path + args.checkpoint_base + args.model_base
    print("Loading", total_path)
    checkpoint = torch.load(total_path, map_location=device)
    new_args = checkpoint["args"]

    if "multiple_encoder" not in new_args:
        new_args.multiple_encoder = False
    if "multiple_hyperprior" not in new_args:
        new_args.multiple_hyperprior = False
    if "delta_encode" not in new_args:
        new_args.delta_encode = False 
    if "residual_before_lrp" not in new_args:
        new_args.residual_before_lrp = False 
    if "double_dim" not in new_args:
        new_args.double_dim = False



    lmbda_list = new_args.lmbda_list
    base_net = get_model(new_args,device, lmbda_list)

    checkpoint_model = replace_keys(checkpoint["state_dict"],multiple_encoder=new_args.multiple_encoder)
    base_net.load_state_dict(checkpoint_model ,strict = True) 

    base_net = base_net.to(device)
    base_net.update() 

    net = models["post"](base_net,
                        N = args.post_N,
                        M = args.post_M,
                        post = args.post)
    net = net.to(device)

    print("MODELLO INIZIALIZZATO! initialize dataset")
    
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([ transforms.ToTensor()])

    train_dataset = ImageFolder(openimages_path, 
                                split="train", 
                                transform=train_transforms, 
                                num_images=args.num_images)

    valid_dataset = ImageFolder(openimages_path, 
                                split="test",
                                 transform=train_transforms, 
                                num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir=kodak_path, transform= test_transforms)

    filelist = test_dataset.image_path


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

    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.test_batch_size, 
                                num_workers=args.num_workers,
                                shuffle=False,
                                pin_memory=(device == "cuda"),)


    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    last_epoch = 0
    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50,70], gamma=0.1, last_epoch=-1) 


    criterion = DistortionLoss() 
    best_loss = float("inf")
    counter = 0
    epoch_enc = 0
    mask_pol  = args.mask_pol
    list_quality_training = args.list_quality_training


    if args.post:
        net.freeze() #freeze everything but post 
    else:
        net.freeze()
        net.unfreeze_g_s()
    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()

        num_tainable = net.print_information()
        
        if num_tainable > 0:
            counter = train_one_epoch( net, 
                                      criterion, 
                                      train_dataloader, 
                                      optimizer, 
                                      epoch, 
                                      counter,
                                      mask_pol ,
                                      list_quality_training,
                                      sampling_training = args.sampling_training)
            
        print("finito il train della epoca")
        pr_list = list_quality_training 
        loss = valid_epoch(epoch, 
                           valid_dataloader,
                           criterion, 
                           net, 
                           pr_list = pr_list,
                           mask_pol=mask_pol
                           )
        print("finito il valid della epoca")
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        lr_scheduler.step()
        list_pr = [0.01,0.1,0.5,1,3,5,7,10]


        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       net, 
                       pr_list = list_pr ,  
                       mask_pol = mask_pol)
        print("finito il test della epoca: ",bpp_t," ",psnr_t)


        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


        bpp, psnr,_ = compress_with_ac(net,  
                                        filelist, 
                                        device,
                                        epoch = epoch_enc, 
                                        pr_list =[0,0.05,0.075,0.1,0.125], 
                                        post = True, 
                                        mask_pol = mask_pol,
                                        cheating = True)

        if args.post:
            bpp_f, psnr_f,_ = compress_with_ac(net,  
                                            filelist, 
                                            device,
                                            epoch = epoch_enc, 
                                            pr_list =[0,0.05,0.075,0.1,0.125], 
                                            post = False, 
                                            mask_pol = mask_pol,
                                            cheating = True)
        psnr_res = {}
        bpp_res = {}

        bpp_res["our_post"] = bpp
        psnr_res["our_post"] = psnr

        if args.post:
            bpp_res["our_f"] = bpp_f
            psnr_res["our_f"] = psnr_f

        #psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
        #bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

        bpp_res["tri_planet_23"] = tri_planet_23_bpp
        psnr_res["tri_planet_23"] = tri_planet_23_psnr

        bpp_res["tri_planet_22"] = tri_planet_22_bpp
        psnr_res["tri_planet_22"] = tri_planet_22_psnr

        plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="compression")

        stringa_lambda = ""
        for lamb in new_args.lmbda_list:
            stringa_lambda = stringa_lambda  + "_" + str(lamb)


        name_folder = args.code + "_" + stringa_lambda + "_"
        
        
        cartella = os.path.join(save_path,name_folder) #dddd


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
                    "state_dict_base": net.base_net.state_dict(),
                    "state_dict_post":net.post_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer": "None",
                    "args":args,
      
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
