import math 
import torch.nn as nn 
import torch 

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))for likelihoods in output["likelihoods"].values())
        
        
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
    

class ScalableRateDistortionLoss(nn.Module):

    def __init__(self, weight = 1, lmbda_starter = 0.75, scalable_levels = 5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.scalable_levels = scalable_levels
        self.lmbda_starter = lmbda_starter
        self.lmbda =  [self.lmbda_starter*(2**(-i)) for i in range(self.scalable_levels)][::-1]
        print("******************  --->",self.lmbda)
        self.scales_tensor = torch.tensor(self.lmbda).view(-1, 1, 1, 1) # (5, 1, 1, 1)
        self.weight = weight

        

    def forward(self,output,target): 

        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W




        # Check the batch sizes of images and recon_images
        batch_size_recon = output["x_hat"].shape[0] # N * num_levels

        batch_size_images =  N
        #print(batch_size_images,"  ",N)

        # If the batch sizes of images and recon_images are different, adjust the batch size
        if batch_size_images != batch_size_recon:
            # Copy images to match the batch size of recon_images
            rate = batch_size_recon // batch_size_images  # rate = 5
            
            extend_images = torch.cat([target] * rate, dim=0)

        scales = self.scales_tensor.repeat((batch_size_images, 1, 1, 1)).to(target.device)   # (batch_size * perce, 1
        out["mse_loss"] = ((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight



        out["bpp_loss_hype"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) 
                                   for likelihoods in output["likelihoods_hyperprior"].values())


        log_y_likelihoods = torch.log(output["likelihoods"]["y"])
        log_residual_likelihoods = torch.log(output["likelihoods"]["r"])   

        out["bpp_loss_y"]  = torch.sum(log_y_likelihoods) / (-math.log(2) * num_pixels)
        out["bpp_loss_res"] = torch.sum(log_residual_likelihoods) / (-math.log(2) * num_pixels)

        out["bpp_loss"] = out["bpp_loss_hype"] + out["bpp_loss_y"] + out["bpp_loss_res"]

        out["loss"] = out["bpp_loss"] + out["mse_loss"] 
        return out

