import torch 
import wandb
from compress.utils.functions import AverageMeter, read_image
import math 
from pytorch_msssim import ms_ssim
import torch.nn.functional as F 
from compressai.ops import compute_padding
from compress.utils.functions import compute_msssim, compute_psnr

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

import random
def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, counter,clip_max_norm = 1.0):
    model.train()
    device = next(model.parameters()).device


    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    bpp_scalable = AverageMeter()
    bpp_main = AverageMeter()


    lmbda_list = model.lmbda_list


    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        #quality =  random.randint(0, len(lmbda_list) - 1)
        #lmbda = lmbda_list[quality]

        out_net = model(d,training = True)

        #out_criterion = criterion(out_net, d, lmbda = lmbda)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()


        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()



        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].mean().clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())
        bpp_scalable.update(out_criterion["bpp_scalable"].clone().detach())
        bpp_main.update(out_criterion["bpp_base"].clone().detach())


        wand_dict = {
            "train_batch": counter,
            "train_batch/loss": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
            "train_batch/bpp_base":out_criterion["bpp_base"].clone().detach().item(),
            "train_batch/bpp_progressive":out_criterion["bpp_scalable"].clone().detach().item(),


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
                f"\tAux loss: {aux_loss.item():.2f}"
            )

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/bpp_progressive":bpp_scalable.avg,
        "train/bpp_base":bpp_main.avg,

        }
        
    wandb.log(log_dict)
    return counter

import torch.nn as nn
def criterion_test(output,target):
    mse = nn.MSELoss()
    N, _, H, W = target.size()
    out = {}
    num_pixels = N * H * W



    out["bpp_hype"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods_hyperprior"].values())
    out["bpp_y"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())
        
    out["bpp_loss"] = out["bpp_y"] + out["bpp_hype"]
    out["mse_loss"] = mse(output["x_hat"], target)


    return out
    


def valid_epoch(epoch, test_dataloader,criterion, model, pr_list = [0.05, 0.04, 0.03,0.02,0.01]):
    #pr_list =  [0] +  pr_list  + [-1]
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter() 
    bpp_loss =AverageMeter() 
    mse_loss = AverageMeter() 
    
    psnr = AverageMeter() 

    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)

            for _,p in enumerate(pr_list):


                quality =  p
                lmbda = model.lmbda_list[quality]
                        
                out_net = model(d, quality = p, training = False)

  
                out_criterion = criterion(out_net, d, lmbda = lmbda)
                psnr_im = compute_psnr(d, out_net["x_hat"])

                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"].clone().detach())
                mse_loss.update(out_criterion["mse_loss"].mean())
                psnr.update(psnr_im)
                    
    log_dict = {
            "valid":epoch,
            "valid/loss": loss.avg,
            "valid/bpp":bpp_loss.avg,
            "valid/mse": mse_loss.avg,
            "valid/psnr":psnr.avg,

         #   "test/y_loss_"+ name[i]: y_loss[i].avg,
            }

    wandb.log(log_dict)
    return loss.avg


def test_epoch(epoch, test_dataloader,criterion, model, pr_list = [0.05, 0.04, 0.03,0.02,0.01], mask_pol = None):
    model.eval()
    device = next(model.parameters()).device


    bpp_loss =[AverageMeter()  for _ in range(len(pr_list) + 2)] 
    psnr = [AverageMeter()  for _ in range(len(pr_list) + 2)]


    #pr_list =  [0] +  pr_list  + [-1]

    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)

            for j,p in enumerate(pr_list):


                quality =  p
                lmbda = model.lmbda_list[quality]
                        
                out_net = model(d, training = False, quality =  p, mask_pol = mask_pol)

  
                out_criterion = criterion(out_net, d, lmbda = lmbda)

                psnr_im = compute_psnr(d, out_net["x_hat"])
                psnr[j].update(psnr_im)

                bpp_loss[j].update(out_criterion["bpp_loss"])




    for i in range(len(pr_list)):
        if i== 0:
            name = "test_base"
        elif i == len(pr_list) - 1:
            name = "test_complete"
        else:
            c = str(i)
            name = "test_quality_" + c 
        
        log_dict = {
            name:epoch,
            name + "/bpp":bpp_loss[i].avg,
            name + "/psnr":psnr[i].avg,
            }

        wandb.log(log_dict)
    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))]


def compress_with_ac(model,  filelist, device, epoch, pr_list,   writing = None, mask_pol = None):
    #pr_list = [0] + pr_listtes + [-1]
    #model.update(None, device)
    l = len(pr_list)
    print("ho finito l'update")
    bpp_loss = [AverageMeter() for _ in range(l)]
    psnr =[AverageMeter() for _ in range(l)]
    mssim = [AverageMeter() for _ in range(l)]

    with torch.no_grad():
        for i,d in enumerate(filelist):
            #print("******************************* image ",d," ***************************************") 



            x = read_image(d).to(device)
            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            for j,p in enumerate(pr_list):
                print("******************************* ",p)

                name = "level_" + str(j)

                data =  model.compress(x_padded, quality =p, mask_pol = mask_pol )
                out_dec = model.decompress(data["strings"], data["shape"], quality = p, mask_pol = mask_pol)

                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)     

                psnr_im = compute_psnr(x, out_dec["x_hat"])
                ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
                ms_ssim_im = -10*math.log10(1 - ms_ssim_im )
                psnr[j].update(psnr_im)
                mssim[j].update(ms_ssim_im)
            
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]

                # calcolo lo stream del base 
                data_string = data["strings"]#[:2]
                bpp = sum(len(s[0]) for s in data_string) * 8.0 / num_pixels

                """
                if p in list(model.lmbda_index_list.keys()):
                    q = model.lmbda_index_list[p] 
                else:
                    q = p

                if q != 0:
                    data_string_hype = data["strings"][2]
                    bpp_hype = sum(len(s) for s in data_string_hype) * 8.0 / num_pixels

                    data_string_scale = data["strings"][-1] # questo è una lista
                    bpp_scale = sum(len(s) for s in data_string_scale) * 8.0 / num_pixels #ddddddd

                    bpp += bpp_scale 
                    bpp += bpp_hype
                """

                bpp_loss[j].update(bpp)

                
                #print("quality: ",p," ",bpp," ",psnr_im)

                if writing is not None:
                    fls = writing + "/" +  name + "/_" +  str(epoch) +  "_.txt"
                    f=open(fls , "a+")
                    f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                    f.close()  

    if epoch > -1:
        for i in range(len(pr_list)):
            if i== 0:
                name = "compress_base"
            elif i == len(pr_list) - 1:
                name = "compress_complete"
            else:
                c = str(i)
                name = "compress_quality_" + c
            
            log_dict = {
                name:epoch,
                name + "/bpp":bpp_loss[i].avg,
                name + "/psnr":psnr[i].avg,
                }

            wandb.log(log_dict)


    #print("enhanced compression results : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    if writing is not None:

        fls = writing + "/" +  name + "/_" +  str(epoch) +  "_.txt"
        f=open(fls , "a+")
        f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss.avg) + " YPSNR " +  str(psnr.avg)  + " YMSSIM " +  str(mssim.avg) + "\n")
    
    
    
    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))]