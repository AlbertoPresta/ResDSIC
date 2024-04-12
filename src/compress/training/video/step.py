import torch 
import wandb
from compress.utils.functions import AverageMeter, read_image
import math 
import random
import torch.nn as nn
from torch.nn.functional import mse_loss
from pytorch_msssim import ms_ssim
import torch.nn.functional as F 
from compressai.ops import compute_padding
from compress.utils.functions import compute_msssim, compute_psnr
import torch.nn.functional as F

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_aux_loss(aux_list, backward=False):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss

        if backward is True:
            aux_loss.backward()

    return aux_loss_sum


def train_one_epoch(counter,
                    model, 
                    criterion, 
                    train_dataloader, 
                    optimizer, 
                    aux_optimizer,
                    epoch, 
                    clip_max_norm = 1.0,
                    scalable = False):
    model.train()
    device = next(model.parameters()).device


    mse_l = AverageMeter()
    bpp_l = AverageMeter()
    bpp_l_keyframe = AverageMeter()
    bpp_l_motion = AverageMeter()
    bpp_l_residual = AverageMeter()
    loss = AverageMeter()

    if scalable:
        mse_l_base = AverageMeter()
        mse_l_prog = AverageMeter()
        bpp_l_base = AverageMeter()
        bpp_l_prog = AverageMeter()
        bpp_l_keyframe_prog = AverageMeter()
        bpp_l_motion_prog = AverageMeter()
        bpp_l_residual_prog = AverageMeter()       

    for i, batch in enumerate(train_dataloader):

        d = [frames.to(device) for frames in batch]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        wand_dict = {
            "train_batch": counter,
            "train_batch/loss": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/bpp_base":out_criterion["bpp_base"].mean().clone().detach().item(),
            "train_batch/bpp_prog":out_criterion["bpp_prog"].mean().clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
        }
        wandb.log(wand_dict)

        bpp_k = out_criterion["bpp_info_dict"]["bpp_loss.keyframe"].mean().clone().detach().item()
        bpp_m = out_criterion["bpp_info_dict"]["bpp_loss.motion"].mean().clone().detach().item()
        bpp_r = out_criterion["bpp_info_dict"]["bpp_loss.residual"].mean().clone().detach().item()
        
        bpp_l_keyframe.update(bpp_k)
        bpp_l_motion.update(bpp_m)
        bpp_l_residual.update(bpp_r)
        mse_l.update(out_criterion["mse_loss"].mean().clone().detach().item())
        bpp_l.update( out_criterion["bpp_loss"].mean().clone().detach().item())
        loss.update(out_criterion["loss"].mean().clone().detach().item())
        
        wand_dict = {
            "train_batch": counter,
            "train_batch/bpp_keyframe":bpp_k,
            "train_batch/bpp_motion": bpp_m,
            "train_batch/bpp_residual":bpp_r,
        }
        wandb.log(wand_dict)


        if scalable:
            bpp_k_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.keyframe"].clone().detach().item()
            bpp_m_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.motion"].clone().detach().item()
            bpp_r_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.residual"].clone().detach().item()
            
            bpp_l_keyframe.update(bpp_k)
            bpp_l_motion.update(bpp_m)
            bpp_l_keyframe_prog.update(bpp_k_prog)
            bpp_l_motion_prog.update(bpp_m_prog)
            bpp_l_residual_prog.update(bpp_r_prog)
            mse_l_base.update(out_criterion["mse_base"].mean().clone().detach().item())
            mse_l_prog.update(out_criterion["mse_prog"].mean().clone().detach().item())
            bpp_l_prog.update(out_criterion["bpp_prog"].clone().detach().item())
            bpp_l_base.update(out_criterion["bpp_base"].clone().detach().item())

        counter += 1

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        aux_optimizer.step()

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
        "train/bpp": bpp_l.avg,
        "train/mse": mse_l.avg,
        "train/bpp_keyframe":bpp_l_keyframe.avg,
        "train/bpp_motion":bpp_l_motion.avg,
        "train/bpp_residual_base":bpp_l_residual.avg
        }
    wandb.log(log_dict)

    if scalable:


        psnr_base = -10 * math.log10(mse_l_base.avg)
        psnr_prog = -10 * math.log10(mse_l_prog.avg)
        log_dict = {
            "train":epoch,
            "train/mse_base":mse_l_base.avg,
            "train/mse_prog":mse_l_prog.avg,
            "train/psnr_base":psnr_base,
            "train/psnr_prog":psnr_prog,
            "train/bpp_base":bpp_l_base.avg,
            "train/bpp_prog":bpp_l_prog.avg,
            "train/bpp_motion_prog":bpp_l_motion_prog.avg,
            "train/bpp_residual_prog":bpp_l_residual_prog.avg

            }
        wandb.log(log_dict)   

    return counter

def valid_epoch(epoch, test_dataloader, model, criterion,scalable = False):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    bpp_l_keyframe = AverageMeter()
    bpp_l_motion = AverageMeter()
    bpp_l_residual = AverageMeter()

    if scalable: 
        bpp_l_keyframe_prog = AverageMeter()
        bpp_l_motion_prog = AverageMeter()
        bpp_l_residual_prog = AverageMeter() 
        bpp_l_base = AverageMeter()
        bpp_l_prog = AverageMeter()  
        mse_l_base = AverageMeter()
        mse_l_prog = AverageMeter() 

  
    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])


            bpp_k = out_criterion["bpp_info_dict"]["bpp_loss.keyframe"].clone().detach().item()
            bpp_m = out_criterion["bpp_info_dict"]["bpp_loss.motion"].clone().detach().item()
            bpp_r = out_criterion["bpp_info_dict"]["bpp_loss.residual"].clone().detach().item()
            
            bpp_l_keyframe.update(bpp_k)
            bpp_l_motion.update(bpp_m)
            bpp_l_residual.update(bpp_r)


            if scalable:
                bpp_k_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.keyframe"].clone().detach().item()
                bpp_m_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.motion"].clone().detach().item()
                bpp_r_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.residual"].clone().detach().item()

                bpp_l_keyframe_prog.update(bpp_k_prog)
                bpp_l_motion_prog.update(bpp_m_prog)
                bpp_l_residual_prog.update(bpp_r_prog)

                bpp_l_base.update(out_criterion["bpp_base"])
                bpp_l_prog.update(out_criterion["bpp_prog"])

                mse_l_base.update(out_criterion["mse_base"].mean().clone().detach().item())
                mse_l_prog.update(out_criterion["mse_prog"].mean().clone().detach().item())

    print(
        f"valid epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    log_dict = {
            "valid":epoch,
            "valid/loss": loss.avg,
            "valid/bpp":bpp_loss.avg,
            "valid/mse": mse_loss.avg,
            "valid/bpp_keyframe":bpp_l_keyframe.avg,
            "valid/bpp_motion":bpp_l_motion.avg,
            "valid/bpp_residual_base":bpp_l_residual.avg,
            }

    wandb.log(log_dict)

    if scalable:

        psnr_base = -10*math.log10(mse_l_base.avg)
        psnr_prog = -10*math.log10(mse_l_prog.avg)
        log_dict = {
                "valid":epoch,
                "valid/bpp_base":bpp_l_base.avg,
                "valid/bpp_prog":bpp_l_prog.avg,
                "valid/bpp_residual_prog":bpp_l_residual_prog.avg,
                "valid/bpp_keyframe_prog":bpp_l_keyframe_prog.avg,
                "valid/bpp_motion_prog":bpp_l_motion_prog.avg,
                "valid/mse_base":mse_l_base.avg,
                "valid/mse_prog":mse_l_prog.avg,
                "valid/psnr_base":psnr_base.avg,
                "valid/psnr_prog":psnr_prog.avg


                }
        wandb.log(log_dict)  
    return loss.avg





def test_epoch(epoch, test_dataloader, model, criterion,scalable = False):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_l = AverageMeter()
    bpp_l_keyframe = AverageMeter()
    bpp_l_motion = AverageMeter()
    bpp_l_residual = AverageMeter()

    if scalable:
        psnr_l_prog = AverageMeter()
        bpp_l_prog = AverageMeter()
        bpp_l_base = AverageMeter()
        bpp_l_keyframe_prog = AverageMeter()
        bpp_l_motion_prog = AverageMeter()
        bpp_l_residual_prog = AverageMeter()


    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            mse_loss_base = out_criterion["mse_base"].mean().clone().detach().item()
            

            psnr = -10*math.log10(mse_loss_base)
            #psnr = compute_psnr_frames(out_net["x_hat"][0],d)
            psnr_l.update(psnr)
            if scalable:
                mse_loss_prog =out_criterion["mse_prog"].mean().clone().detach().item()
                psnr = -10*math.log10(mse_loss_prog)
                #psnr = compute_psnr_frames(out_net["x_hat"][1],d)
                psnr_l_prog.update(psnr)        

            bpp_k = out_criterion["bpp_info_dict"]["bpp_loss.keyframe"].clone().detach().item()
            bpp_m = out_criterion["bpp_info_dict"]["bpp_loss.motion"].clone().detach().item()
            bpp_r = out_criterion["bpp_info_dict"]["bpp_loss.residual"].clone().detach().item()
            
            bpp_l_keyframe.update(bpp_k)
            bpp_l_motion.update(bpp_m)
            bpp_l_residual.update(bpp_r)

            if scalable:
                bpp_k_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.keyframe"].clone().detach().item()
                bpp_m_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.motion"].clone().detach().item()
                bpp_r_prog = out_criterion["bpp_info_dict_prog"]["bpp_loss.residual"].clone().detach().item()
                
            
                bpp_l_keyframe_prog.update(bpp_k_prog)
                bpp_l_motion_prog.update(bpp_m_prog)
                bpp_l_residual_prog.update(bpp_r_prog)

                bpp_l_base.update( out_criterion["bpp_base"].clone().detach().item())
                bpp_l_prog.update( out_criterion["bpp_prog"].clone().detach().item())



    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    log_dict = {
            "test":epoch,
            "test/loss": loss.avg,
            "test/bpp":bpp_loss.avg,
            "test/mse": mse_loss.avg,
            "test/psnr": psnr_l.avg,
            "test/bpp_keyframe":bpp_l_keyframe.avg,
            "test/bpp_motion":bpp_l_motion.avg,
            "test/bpp_residual":bpp_l_residual.avg
            }
    wandb.log(log_dict)
    if scalable: 

        log_dict = {
                "test_base":epoch,
                "test_base/bpp":bpp_l_base.avg,
                "test_base/psnr": psnr_l.avg,
                "test_base/bpp_keyframe":bpp_l_keyframe.avg,
                "test_base/bpp_motion":bpp_l_motion.avg,
                "test_base/bpp_residual":bpp_l_residual.avg
                }
        wandb.log(log_dict) 

        log_dict = {
                "test_prog":epoch,
                "test_prog/bpp":bpp_l_prog.avg,
                "test_prog/psnr": psnr_l_prog.avg,
                "test_prog/bpp_keyframe":bpp_l_keyframe_prog.avg,
                "test_prog/bpp_motion":bpp_l_motion_prog.avg,
                "test_prog/bpp_residual":bpp_l_residual_prog.avg
                }
        wandb.log(log_dict) 




    return loss.avg


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def crop(x, padding):
    return F.pad(x, tuple(-p for p in padding))


def compute_psnr_frames(output,input):

    assert len(output) == len(input)

    psnr_tot = 0
    for i,d in enumerate(input):
        rec = output[i] 
        psnr_i = compute_psnr(rec,d) 
        psnr_tot += psnr_i 
    return psnr_tot/len(input)

