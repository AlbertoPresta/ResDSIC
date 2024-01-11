import torch 
import wandb
from compress.utils.functions import AverageMeter
import math 
from pytorch_msssim import ms_ssim


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, counter,clip_max_norm = 1.0):
    model.train()
    device = next(model.parameters()).device


    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    bpp_hype = AverageMeter()
    bpp_res = AverageMeter()
    bpp_main = AverageMeter()




    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())
        bpp_hype.update(out_criterion["bpp_loss_hype"].clone().detach())
        bpp_res.update(out_criterion["bpp_loss_res"].clone().detach())
        bpp_main.update(out_criterion["bpp_loss_y"].clone().detach())


        wand_dict = {
            "train_batch": counter,
            "train_batch/losss": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),
            "train_batch/bpp_hyperpriors":out_criterion["bpp_loss_hype"].clone().detach().item(),
            "train_batch/bpp_residual":out_criterion["bpp_loss_res"].clone().detach().item(),
            "train_batch/bpp_main":out_criterion["bpp_loss_y"].clone().detach().item(),

        }
        wandb.log(wand_dict)
        counter += 1


        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()



        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/bpp_residual":bpp_res.avg,
        "train/bpp_hyperprior":bpp_hype.avg,
        "train/bpp_main":bpp_main.avg
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
    


def test_epoch(epoch, test_dataloader, model, criterion,scale_p, valid = True):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            if valid is False:
                out_net = model.forward_test(d,scale_p)
                out_criterion = criterion_test(out_net, d)
                psnr_im = compute_psnr(d, out_net["x_hat"])
            else:
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                psnr_im = compute_psnr(d.repeat(criterion.scalable_levels,1,1,1).to(d.device), out_net["x_hat"])


            bpp_loss.update(out_criterion["bpp_loss"])
            if valid is True:
                loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(psnr_im)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )


    name = "scale_l:" + str(scale_p)

    if valid is False:

        log_dict = {
            "test_" + name :epoch,
            "test_" + name + "/bpp_":bpp_loss.avg,
            "test_" + name + "/mse_": mse_loss.avg,
            "test_" + name + "/psnr_":psnr.avg,
            }
    else:

        log_dict = {
            "valid_" + name:epoch,
            "valid_" + name  + "/loss_": loss.avg,
            "valid_" + name  +"/bpp_":bpp_loss.avg,
            "valid_" + name  + "/mse_": mse_loss.avg,
            "valid_" + name  + "/psnr_"+ name:psnr.avg

            }       

        wandb.log(log_dict)




    return loss.avg
