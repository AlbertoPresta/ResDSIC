import math 
import torch.nn as nn 
import torch 
from torch.nn.functional import mse_loss

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, lmbda = None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W


        if lmbda is None:
            lmbda = self.lmbda


        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())
        
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
    

class ScalableRateDistortionLoss(nn.Module):

    def __init__(self, weight = 255**2, lmbda_list = [0.75],  device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight


        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

        # Check the batch sizes of images and recon_images
        batch_size_recon = output["x_hat"].shape[0] # num_levels,N

        if batch_size_recon != 1:
            # Copy images to match the batch size of recon_images
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        else:
            extend_images = target.unsqueeze(0)
        

        if lmbda is  None:
            lmbda = self.lmbda
        else:
            lmbda = torch.tensor(lmbda).to(self.lmbda.device)


        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') # compute the point-wise mse #((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 



        likelihoods = output["likelihoods"]

        out["bpp_hype_base"] = (torch.log(likelihoods["z"]).sum())/denominator
        out["bpp_main_base"] = (torch.log(likelihoods["y"]).sum())/denominator
        out["bpp_base"]= out["bpp_main_base"] + out["bpp_hype_base"]#(torch.log(likelihoods["y"].squeeze(0)).sum() + torch.log(likelihoods["z"]).sum())/denominator


        out["bpp_hype_scale"] = (torch.log(likelihoods["z_prog"]).sum())/denominator 
        out["bpp_main_scale"] = (torch.log(likelihoods["y_prog"]).sum())/denominator 
        out["bpp_scalable"] = out["bpp_main_scale"] + out["bpp_hype_scale"]#(torch.log(likelihoods["y_prog"]).sum() + torch.log(likelihoods["z_prog"]).sum())/denominator 

        out["bpp_loss"] = out["bpp_scalable"] + batch_size_recon*out["bpp_base"]
        out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() 
        return out

