


import torch
import torch.nn as nn



class Mask(nn.Module):

    def __init__(self, mask_policy, scalable_levels,M):
        super().__init__()

        self.mask_policy = mask_policy 
        self.scalable_levels = scalable_levels 
        self.M = M 


        if self.mask_policy == "learnable-mask-gamma":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels - 2, self.M))) #il primo e il base layer, lìultimo è il completo!!!
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M*2, out_channels=self.M, kernel_size=1, stride=1),)
        if self.mask_policy == "learnable-mask-nested":
            self.mask_conv = nn.ModuleList(
                            nn.Sequential(torch.nn.Conv2d(in_channels=self.M*2, 
                                                          out_channels=self.M, 
                                                          kernel_size=1, 
                                                          stride=1),)
                            for _ in range(self.scalable_levels -2)
                            )



    def apply_noise(self, mask, tr):
            if tr:
                mask = mask + (torch.rand_like(mask) - 0.5)
                mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()   
            else:
                mask = torch.round(mask)
            return mask

    def forward(self,scale,scale_prog = None,  pr = 0, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        bs, ch, w,h = shapes

        if mask_pol == "all-one":
            return torch.ones_like(scale).to(scale.device)
        elif mask_pol == "all-zero":
            return torch.zeros_like(scale).to(scale.device)
        elif mask_pol == "point-based-std":
            if pr == self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0.0:
                return torch.zeros_like(scale).to(scale.device)
            

            assert scale is not None 
            if pr > 1:
                pr = pr*0.1
            pr_bis = 1.0 - pr
            scale = scale.ravel()
            quantile = torch.quantile(scale, pr_bis)
            res = scale >= quantile 
            res = res.float()

            #print("original pr: ",pr,"distribution---> ",torch.unique(res,return_counts = True))
            #print("dovrebbero essere soli 1: ",torch.unique(res, return_counts = True))
            return res.reshape(bs,ch,w,h).to(torch.float)
        elif mask_pol == "learnable-mask-gamma":
            
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            

            if pr == self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)

            assert scale_prog is not None 
            scale_input = torch.cat([scale,scale_prog],dim = 1)
            importance_map =  self.mask_conv(scale_input) 

            importance_map = torch.sigmoid(importance_map) 

            index_pr = self.scalable_levels - 1 - pr
            index_pr = int(index_pr)
            #index_pr = pr - 1
            #gamma = torch.sum(torch.stack([self.gamma[j] for j in range(index_pr)]),dim = 0) # più uno l'hom esso in lmbda_index
            gamma = self.gamma[pr - 1][None, :, None, None]
            gamma = torch.relu(gamma) 


            adjusted_importance_map = torch.pow(importance_map, gamma)

            
            return adjusted_importance_map          

        elif mask_pol == "learnable-mask-nested":

            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if pr == self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)
            
            assert scale_prog is not None 
            scale_input = torch.cat([scale,scale_prog],dim = 1)

            importance_map =torch.sigmoid(self.mask_conv[pr - 1](scale_input)) #torch.sum(torch.stack([torch.sigmoid(self.mask_conv[i](scale_input)) for i in range(pr)],dim = 0),dim = 0) 
            #importance_map = torch.sigmoid(importance_map)    

            return importance_map       

        elif mask_pol == "two-levels":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            else:
                return torch.ones_like(scale).to(scale.device)
        
        elif mask_pol == "scalable_res":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            elif pr == self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)
            else:
                lv_tot = [3*32,6*32] 
                c = torch.zeros_like(scale).to(scale.device)
                lv = lv_tot[pr -1]
                c[:,lv:,:,:] = 1.0

                return c.to(scale.device)     
        else:
            raise NotImplementedError()


    