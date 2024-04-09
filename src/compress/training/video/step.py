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

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_aux_loss(aux_list):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
    return aux_loss_sum


def train_one_epoch(counter,
                    model, 
                    criterion, 
                    train_dataloader, 
                    optimizer, 
                    aux_optimizer,
                    epoch, 
                    clip_max_norm = 1.0):
    model.train()
    device = next(model.parameters()).device


    mse_l = AverageMeter()
    bpp_l = AverageMeter()
    loss = AverageMeter()

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
            "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
        }
        wandb.log(wand_dict)
        counter += 1




        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

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
        }
    wandb.log(log_dict)
    

    return counter

def valid_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

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
            "valid/mse": mse_loss.avg
            }

    wandb.log(log_dict)

    return loss.avg





def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_l = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            d = [frames.to(device) for frames in batch]
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(compute_aux_loss(model.aux_loss()))
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            psnr = compute_psnr_frames(out_net["x_hat"],d)

            psnr_l.update(psnr)



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
            "test/psnr": psnr_l.avg
            }
    

    wandb.log(log_dict)


    return loss.avg


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_psnr_frames(output,input):

    assert len(output) == len(input)

    psnr_tot = 0
    for i,d in enumerate(input):
        rec = output[i] 
        psnr_i = compute_psnr(rec,d) 
        psnr_tot += psnr_i 
    return psnr_tot/len(input)

