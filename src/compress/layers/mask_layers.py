


import torch
import torch.nn as nn
from compressai.ops import LowerBound
from compress.layers import conv3x3, subpel_conv3x3
from compress.ops import ste_round

class Mask(nn.Module):

    def __init__(self, mask_policy, scalable_levels,M = 192):
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
                #mask = mask + (torch.rand_like(mask) - 0.5)
                mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()   
            else:
                mask = torch.round(mask)
            return mask

    def forward(self,scale,scale_prog = None,  pr = 0, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        bs, ch, w,h = shapes

        res = torch.zeros_like(scale).to(scale.device)


        if mask_pol == "point-based-std":
            if pr == 10:# self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0.0:
                return torch.zeros_like(scale).to(scale.device)
            

            assert scale is not None 

            
            assert scale is not None 

            
            pr = pr*0.1
            pr_bis = 1.0 - pr
            scale_flatten = scale.reshape(bs,-1).to(scale.device) 
            quantiles = torch.quantile(scale_flatten, q=pr_bis, dim=(1))
            res = torch.where(scale_flatten < quantiles.view(-1, 1),torch.tensor(0.0), scale_flatten).to(scale.device)
            res = res.float()
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





class ChannelMask(nn.Module):

    def __init__(self, 
                 mask_policy, 
                 scalable_levels,
                 dim_chunk, 
                 num_levels, 
                 gamma_bound = 1e-9, 
                 double_dim = False):
        super().__init__()

        self.mask_policy = mask_policy 
        self.scalable_levels = scalable_levels 
        self.quality_list = [i for i in range(self.scalable_levels)]
        self.dim_chunk =dim_chunk 
        self.num_levels = num_levels
        self.double_dim = double_dim
        
        if self.double_dim:
            print("vado qua dove il double dim è raddoppiato!!!")
            self.input_dim = self.dim_chunk*2
        else: 
            self.input_dim = self.dim_chunk


        if self.mask_policy == "learnable-mask-quantile":
            self.mask_conv = nn.ModuleList(nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU()                           
                            ) for _ in range(self.num_levels)
            )

        elif self.mask_policy == "single-learnable-mask-quantile":
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
            
        elif self.mask_policy == "single-learnable-mask-gamma":
            self.gamma = [
                        torch.nn.Parameter(torch.ones((self.scalable_levels - 2, self.dim_chunk))) 
                        for _ in range(self.num_levels)
            ]
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
            
            self.gamma_lower_bound = LowerBound(gamma_bound)


        elif self.mask_policy == "learnable-mask-gamma":
            self.gamma = [
                        torch.nn.Parameter(torch.ones((self.scalable_levels - 2, self.dim_chunk))) 
                        for _ in range(self.num_levels)
            ]
            self.mask_conv = nn.ModuleList(
                    nn.Sequential(torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.dim_chunk, kernel_size=1, stride=1),) for _ in range(self.num_levels)
                    )
            
            self.gamma_lower_bound = LowerBound(gamma_bound)
        
        
        elif self.mask_policy == "learnable-mask-nested":
            self.mask_conv = nn.ModuleList(
                                nn.ModuleList(
                                    nn.Sequential(torch.nn.Conv2d(in_channels=self.input_dim, 
                                    out_channels=self.dim_chunk, 
                                    kernel_size=1, 
                                    stride=1),) for _ in range(self.scalable_levels -2)
                                    ) for _ in range(self.num_levels) )
        elif self.mask_policy == "single-learnable-mask-nested":
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
        
        elif self.mask_policy == "three-levels-learnable":
            print("STO ENTRANDO QUA,DEVO COSTRUIRLA BENE!!!")
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.input_dim),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.input_dim,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.dim_chunk,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 

            print(self.mask_conv)          

    def apply_noise(self, mask, tr):
            if tr:
                mask = ste_round(mask)
                #mask = mask + (torch.rand_like(mask) - 0.5)
                #mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()   
                
            else:
                mask = torch.round(mask)
            return mask

    def forward(self,
                scale,  
                scale_base = None, 
                slice_index = 0,  
                pr = 0, 
                mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        bs, ch, w,h = shapes

        if mask_pol is None: 
            return torch.ones_like(scale).to(scale.device)
        
        if mask_pol == "point-based-std":
            if pr == 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            shapes = scale.shape
            bs, ch, w,h = shapes
            assert scale is not None 
            pr = pr*0.1
            pr_bis = 1.0 - pr
            scale_flatten = scale.reshape(bs,-1).to(scale.device) 
            quantiles = torch.quantile(scale_flatten, q=pr_bis, dim=(1))
            res = torch.where(scale_flatten < quantiles.view(-1, 1),torch.tensor(0.0), scale_flatten).to(scale.device)
            res = res.float()
            return res.reshape(bs,ch,w,h).to(torch.float)

        elif mask_pol == "point-based-double-std":
            if pr == 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            shapes = scale.shape
            bs, ch, w,h = shapes
            assert scale is not None 
            assert scale_base is not None 
            pr = pr*0.1  
            pr = 1 -pr 
            scale = scale.ravel()
            scale_base = scale_base.ravel()
            scale_t = scale + scale_base
            quantile = torch.quantile(scale_t, pr)
            res = scale_t >= quantile 
            return res.reshape(bs,ch,w,h).to(torch.float).to(scale.device)

        elif mask_pol == "two-levels":
            return torch.zeros_like(scale).to(scale.device) if pr == 0 else torch.ones_like(scale).to(scale.device)
        elif mask_pol == "three-levels-std":
            if pr == 0: 
                return torch.zeros_like(scale).to(scale.device)
            elif pr == 2:
                return torch.ones_like(scale).to(scale.device)
            else:
                shapes = scale.shape
                bs, ch, w,h = shapes
                assert scale is not None 
                pr = 0.5  
                scale = scale.ravel()
                quantile = torch.quantile(scale, pr)
                res = scale >= quantile 
                return res.reshape(bs,ch,w,h).to(torch.float).to(scale.device)
        elif mask_pol == "three-levels-learnable":
            if pr == 0: 
                return torch.zeros_like(scale).to(scale.device)
            elif pr == 2:
                return torch.ones_like(scale).to(scale.device)
            else:
                #assert scale_base is not None
                if self.double_dim:
                    importance_map = self.mask_conv(scale_base)
                else:
                    importance_map =  self.mask_conv(scale) 
                return ste_round(importance_map)

               
                
        elif mask_pol == "single-learnable-mask-quantile":
            if pr == 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            pr = pr*0.1  
            pr = 1 - pr
            shapes = scale.shape
            bs, ch, w,h = shapes
            scale_input = torch.cat([scale,scale_base],dim = 1) if self.double_dim else scale
            importance_map =  self.mask_conv(scale_input)


            quantile_tensor = importance_map.clone()
            quantile_tensor = quantile_tensor.ravel()
            quantile = torch.quantile(quantile_tensor, pr)
            res = quantile_tensor >= quantile
            res = res.reshape(bs,ch,w,h).to(scale.device)

            importance_map = importance_map*res
            
            
            return importance_map

        elif mask_pol == "learnable-mask-quantile":
            pr = pr*0.1  
            pr = 1 - pr
            shapes = scale.shape
            bs, ch, w,h = shapes
            scale_input = torch.cat([scale,scale_base],dim = 1) if self.double_dim else scale
            importance_map =  self.mask_conv[slice_index](scale_input) 

            importance_map = importance_map.ravel()
            quantile = torch.quantile(importance_map, pr)
            res = importance_map >= quantile 
            return res.reshape(bs,ch,w,h).to(torch.float).to(scale.device)


        elif mask_pol == "single-learnable-mask-gamma":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if pr >= self.scalable_levels - 1: #mettiamo
                return torch.ones_like(scale).to(scale.device)

            
            scale_input = torch.cat([scale,scale_base],dim = 1) if self.double_dim else scale
            importance_map =  self.mask_conv(scale_input) 

            #importance_map = torch.sigmoid(importance_map) 
            gamma = self.gamma[slice_index][pr - 1][None, :, None, None].to(scale.device)
            gamma = torch.relu(gamma) 
            gamma = self.gamma_lower_bound(gamma)
            adjusted_importance_map = torch.pow(importance_map, gamma)
    
            return adjusted_importance_map    


        elif mask_pol == "learnable-mask-gamma":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if self.scalable_levels - 1: #mettiamo
                return torch.ones_like(scale).to(scale.device)


            scale_input = torch.cat([scale,scale_base],dim = 1) if self.double_dim else scale
            importance_map =  self.mask_conv[slice_index](scale_input) 

            importance_map = torch.sigmoid(importance_map) 
            gamma = self.gamma[slice_index][pr - 1][None, :, None, None].to(scale.device) #dddd
            gamma = torch.relu(gamma) 
            gamma = self.gamma_lower_bound(gamma)


            adjusted_importance_map = torch.pow(importance_map, gamma)

            
            return adjusted_importance_map          

        elif mask_pol == "learnable-mask-nested":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if pr >= self.scalable_levels - 1:
                return torch.ones_like(scale).to(scale.device)

            scale_input = torch.cat([scale,scale_base],dim = 1) if self.double_dim else scale 
            importance_map =torch.sum(torch.stack([torch.sigmoid(self.mask_conv[i](scale_input)) for i in range(pr)],dim = 0),dim = 0) 
  
            return importance_map   

        elif mask_pol == "random":
            shape = scale.shape
            pr = pr*10
            total_elements = torch.prod(torch.tensor(shape)).item()
            num_ones = int(total_elements * (pr / 100.0))
            tensor = torch.zeros(shape).to(scale.device)
            indices = torch.randperm(total_elements)[:num_ones]
            # Impostiamo gli elementi selezionati su 1
            tensor.view(-1)[indices] = 1  

            #print("percentage random: ",(total_elements - num_ones)/total_elements)
            return tensor   
        elif mask_pol == "scalable_res":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            elif pr ==10:
                return torch.ones_like(scale).to(scale.device)
            else:
                pr = pr*0.1
                ones_channel = int(320*pr) 
                canale_inizio = slice_index*32
                canale_fine = 32*(slice_index + 1)
                if ones_channel >= canale_fine: 
                    c = torch.ones_like(scale).to(scale.device)
                    return c 
                elif ones_channel < canale_inizio:
                    c = torch.zeros_like(scale).to(scale.device)
                    return c 
                else: 
                    c = torch.zeros_like(scale).to(scale.device)
                    remainder = ones_channel%32
                    c[:,remainder:,:,:] = 1 
                    return c.to(scale.device)

        else:
            raise NotImplementedError()