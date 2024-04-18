import math 
import torch.nn as nn 
import torch 
from torch.nn.functional import mse_loss

class MaskRateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2,weight = 255**2, device = "cuda"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.weight = weight
        self.device = device

    def forward(self,output,target,lmbda = None): 

        lmbda = self.lmbda if lmbda is None else torch.tensor([lmbda]).to(self.device) #dddd

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

        #batch_size_recon = output["x_hat"].shape[0] # num_levels,N
        #if batch_size_recon != 1:
            # Copy images to match the batch size of recon_images
        #    target = target.unsqueeze(0)
        #    extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        #else:
        #    extend_images = target.unsqueeze(0)
        

        out["mse_loss"] = self.mse(output["x_hat"][-1].squeeze(0), target)
        #out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') # compute the point-wise mse #((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight
        #out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


        denominator = -math.log(2) * num_pixels  
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator

        if "y_prog" in list(likelihoods.keys()):
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_hype"]
        else: 
            out["bpp_base"] = (torch.log(likelihoods["y"].squeeze(0)).sum())/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_hype"]
        out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean()
        #out["loss"] = self.weight*(lmbda*out["mse_loss"]).mean()  
        return out



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

    def __init__(self, weight = 255**2, lmbda_list = [0.75],  device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.device = device

        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

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
            lmbda = torch.tensor([lmbda]).to(self.device)

  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') # compute the point-wise mse #((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = (torch.log(likelihoods["y"].squeeze(0)).sum())/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() 
        return out




class ScalableDistilledRateDistortionLoss(nn.Module):

    def __init__(self, 
                 encoder_enhanced,
                 encoder_base = None,
                 weight = 255**2, 
                 lmbda_list = [0.75],
                 gamma = 0.5,
                device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.weight_kd = 1
        self.device = device
        self.encoder_enhanced = encoder_enhanced.to(self.device)
        self.encoder_base = encoder_base
        if self.encoder_base is not None:
            self.encoder_base = self.encoder_base.to(self.device) 
        self.gamma = gamma

        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W


        # first of all 
        #print("come prima cosa valutiamo il rapporto tra mse e kd, se sono sulla stessa scala")
        #print("target shape: ",target.shape)
        y_kd_enhanced = self.encoder_enhanced(target).to(self.device)
        kd_enh = mse_loss(output["y_hat"][1],y_kd_enhanced)*self.weight_kd
        out["kd_enh"] = kd_enh


        if self.encoder_base is not None:
            y_kd_base = self.encoder_base(target).to(self.device)
            kd_base = mse_loss(output["y_hat"][0],y_kd_base)*self.weight_kd
            out["kd_base"] = kd_base


        batch_size_recon = output["x_hat"].shape[0] # num_levels,N
        if batch_size_recon != 1:
            # Copy images to match the batch size of recon_images #ssss
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        else:
            extend_images = target.unsqueeze(0)
        
        if lmbda is  None:
            lmbda = self.lmbda
        else:
            lmbda = torch.tensor([lmbda]).to(self.device)
  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') 
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


        
        

        
        

        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator#(1 - self.gamma)*out["kd_base"] + self.gamma*((torch.log(likelihoods["y"]).sum())/denominator)
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator#(1 - self.gamma)*out["kd_enh"] + self.gamma*((torch.log(likelihoods["y_prog"]).sum()).sum()/denominator) 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = (1 - self.gamma)*kd_enh + self.gamma*(torch.log(likelihoods["y"].squeeze(0)).sum())/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])

        out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() + out["kd_enh"]*(lmbda[-1]*0.5)
        if self.encoder_base is not None:
            out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() \
                        + out["kd_enh"]*(lmbda[-1]*self.gamma) + out["kd_base"]*(lmbda[0]*self.gamma)
        else:
            out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean()\
                         + out["kd_enh"]*(lmbda[-1]*self.gamma)
        return out
    



class DistilledRateLoss(nn.Module):

    def __init__(self, 
                 encoder_enhanced,
                 encoder_base = None,
                 weight = 255**2, 
                 lmbda_list = [0.75],

                device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.weight_kd = 1
        self.device = device
        self.encoder_enhanced = encoder_enhanced.to(self.device)
        self.encoder_base = encoder_base
        if self.encoder_base is not None:
            self.encoder_base = self.encoder_base.to(self.device) 


        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

        y_kd_enhanced = self.encoder_enhanced(target).to(self.device)
        kd_enh = mse_loss(output["y_hat"][1],y_kd_enhanced)*self.weight_kd
        out["kd_enh"] = kd_enh


        if self.encoder_base is not None:
            y_kd_base = self.encoder_base(target).to(self.device)
            kd_base = mse_loss(output["y_hat"][0],y_kd_base)*self.weight_kd
            out["kd_base"] = kd_base


        batch_size_recon = output["x_hat"].shape[0] # num_levels,N
        if batch_size_recon != 1:
            # Copy images to match the batch size of recon_images #ssss
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        else:
            extend_images = target.unsqueeze(0)
        
        if lmbda is  None:
            lmbda = self.lmbda
        else:
            lmbda = torch.tensor([lmbda]).to(self.device)
  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') 
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


    
        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator#(1 - self.gamma)*out["kd_base"] + self.gamma*((torch.log(likelihoods["y"]).sum())/denominator)
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator#(1 - self.gamma)*out["kd_enh"] + self.gamma*((torch.log(likelihoods["y_prog"]).sum()).sum()/denominator) 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = torch.log(likelihoods["y"].squeeze(0)).sum()/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])


        out["loss"] = out["bpp_loss"] + self.weight*(lmbda[-1]*out["kd_enh"])

        return out
    


class ScalableDistilledDistortionLoss(nn.Module):

    def __init__(self, 
                 encoder_enhanced,
                 encoder_base = None,
                 weight = 255**2, 
                 lmbda_list = [0.75],
                 gamma = 0.5,
                device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.weight_kd = 1
        self.device = device
        self.encoder_enhanced = encoder_enhanced.to(self.device)
        self.encoder_base = encoder_base
        if self.encoder_base is not None:
            self.encoder_base = self.encoder_base.to(self.device) 
        self.gamma = gamma

        

    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W
        # first of all 
        #print("come prima cosa valutiamo il rapporto tra mse e kd, se sono sulla stessa scala")
        #print("target shape: ",target.shape)
        y_kd_enhanced = self.encoder_enhanced(target).to(self.device)
        kd_enh = mse_loss(output["y_hat"][1],y_kd_enhanced)*self.weight_kd
        out["kd_enh"] = kd_enh


        if self.encoder_base is not None:
            y_kd_base = self.encoder_base(target).to(self.device)
            kd_base = mse_loss(output["y_hat"][0],y_kd_base)*self.weight_kd
            out["kd_base"] = kd_base


        batch_size_recon = output["x_hat"].shape[0] # num_levels,N
        if batch_size_recon != 1:
            # Copy images to match the batch size of recon_images #ssss
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) #num_levels, BS,W,H
        else:
            extend_images = target.unsqueeze(0)
        
        if lmbda is  None:
            lmbda = self.lmbda
        else:
            lmbda = torch.tensor([lmbda]).to(self.device)
  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') 
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


    
        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator#(1 - self.gamma)*out["kd_base"] + self.gamma*((torch.log(likelihoods["y"]).sum())/denominator)
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator#(1 - self.gamma)*out["kd_enh"] + self.gamma*((torch.log(likelihoods["y_prog"]).sum()).sum()/denominator) 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = torch.log(likelihoods["y"].squeeze(0)).sum()/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])

        #out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() + out["kd_enh"]*(lmbda[-1]*0.5)
        if self.encoder_base is not None:
            out["loss"] = self.gamma*(out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean()) \
                        + (1 - self.gamma)*(out["kd_enh"] + out["kd_base"])
        else:
            out["loss"] = self.gamma*(out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean())\
                         + out["kd_enh"]*(1 - self.gamma)
        return out
    



class DistortionLoss(nn.Module):

    def __init__(self, weight = 255**2,  device = "cuda"):
        super().__init__()
        self.weight = weight
        self.device = device
        

    def forward(self,output,target): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

        batch_size_recon = output["x_hat"].shape[0] # num_levels,N

        if batch_size_recon != 1:
            target = target.unsqueeze(0)
            extend_images = target.repeat(batch_size_recon,1,1,1,1) 
        else:
            extend_images = target.unsqueeze(0)


        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none')
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) 


        denominator = -math.log(2) * num_pixels 
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator


        if "y_prog" in list(likelihoods.keys()):
            
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator#(1 - self.gamma)*out["kd_base"] + self.gamma*((torch.log(likelihoods["y"]).sum())/denominator)
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator#(1 - self.gamma)*out["kd_enh"] + self.gamma*((torch.log(likelihoods["y_prog"]).sum()).sum()/denominator) 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = torch.log(likelihoods["y"].squeeze(0)).sum()/denominator
            #out["bpp_scalable"] = (torch.log(likelihoods["y"][:,1:,:,:,:].squeeze(0)).sum())/denominator
            out["bpp_loss"] =  out["bpp_base"] + batch_size_recon*(out["bpp_hype"]) # + out["bpp_scalable"] 

        out["loss"] = self.weight*(out["mse_loss"]).mean() 
        return out


##########################################################################################
##########################################################################################
##########################################################################################



class ScalableMutualRateDistortionLoss(nn.Module):

    def __init__(self, weight = 255**2, lmbda_list = [0.75],gamma = 0,  device = "cuda"):
        super().__init__()
        self.scalable_levels = len(lmbda_list)
        self.lmbda = torch.tensor(lmbda_list).to(device) 
        self.weight = weight
        self.device = device
        self.gamma = gamma



    def compute_pearson(self,x,y,mu,std, eps = 1e-6):
        #assert isistance(mu,List)
        #assert isistance(std,List)

        vx = x - mu[0]
        vy = y - mu[1]
        numerator = torch.sum(vx*vy)
        #denominator = (torch.norm(vx**2)*torch.norm(vy**2)) + eps
        denominator = (std[0]*std[1]) + eps
        
        pearson =  numerator/denominator
        return torch.abs(pearson.mean())


    def compute_mutual(self,pears_corr, eps = 1e-7):
        argument = 1 - pears_corr**2 + eps 

        print("argument: ",argument)
        return -0.5*math.log(argument)


    def forward(self,output,target,lmbda = None): 

        batch_size_images, _, H, W = target.size()
        out = {}
        num_pixels = batch_size_images * H * W

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
            lmbda = torch.tensor([lmbda]).to(self.device)

  
        out["mse_loss"] = mse_loss(extend_images,output["x_hat"],reduction = 'none') # compute the point-wise mse #((scales * (extend_images - output["x_hat"])) ** 2).mean()*self.weight
        out["mse_loss"] = out["mse_loss"].mean(dim=(1,2,3,4)) #dim = num_levels 


        denominator = -math.log(2) * num_pixels  # (batch_size * perce, 1
        likelihoods = output["likelihoods"]
        out["bpp_hype"] =  (torch.log(likelihoods["z"]).sum())/denominator

        if "z_prog" in list(out.keys()):
            out["bpp_hype"] = out["bpp_hype"] +  torch.log(likelihoods["z_prog"]).sum()/denominator


        if "y_prog" in list(likelihoods.keys()):
            out["bpp_base"] = (torch.log(likelihoods["y"]).sum())/denominator
            out["bpp_scalable"] = (torch.log(likelihoods["y_prog"]).sum()).sum()/denominator 
            out["bpp_loss"] = out["bpp_scalable"] +  out["bpp_base"] + batch_size_recon*(out["bpp_hype"])
        else: 
            out["bpp_base"] = (torch.log(likelihoods["y"].squeeze(0)).sum())/denominator
            out["bpp_scalable"] = ((torch.log(likelihoods["y"]).sum()).sum()/denominator)*0.0
            out["bpp_loss"] = out["bpp_scalable"] + out["bpp_base"] + batch_size_recon*(out["bpp_hype"])


        # compute mutual information

        mu_s = ( torch.cat(output["mu_base"],dim = 1) , torch.cat(output["mu_prog"],dim = 1) )
        std_s = ( torch.cat(output["std_base"],dim = 1) , torch.cat(output["std_prog"],dim = 1) )
        out["pearson"] = self.compute_pearson(output["y_base"],output["y_prog"],mu_s,std_s)

        if self.gamma != 0:
            out["mutual"] = self.compute_mutual(out["pearson"])
            out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean() + out["mutual"]
        else:
            out["mutual"] = self.weight*(lmbda*out["mse_loss"]).mean()*0.0
            out["loss"] = out["bpp_loss"] + self.weight*(lmbda*out["mse_loss"]).mean()            
        return out




