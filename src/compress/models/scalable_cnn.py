import math
import torch
import torch.nn as nn

from compress.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional
from compress.layers import GDN
from .utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
from .base import CompressionModel
from .cnn import WACNN
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ResWACNN(WACNN):
    """CNN based model"""

    def __init__(self, N=192,
                M=320,
                mask_policy = "learnable-mask",
                lmbda_list = [0.03,0.01],
                **kwargs):
        super().__init__(N = N, M = M, **kwargs)

        self.N = N 
        self.M = M 
        self.halve = 8
        self.level = 5 if self.halve == 8 else -1 
        self.factor = self.halve**2
        assert self.N%self.factor == 0 
        self.T = int(self.N//self.factor) + 3
        self.mask_policy = mask_policy
        self.lmbda_list = lmbda_list

        self.scalable_levels = len(self.lmbda_list)

        self.lmbda_index_list = dict(zip(self.lmbda_list[::-1], [i  for i in range(len(self.lmbda_list))] ))
        #self.lmbda_index_list[lmbda_list[0]] = 100  # let the all latent space with the highest lamnda
        
        
        self.entropy_bottleneck_prog = EntropyBottleneck(self.N)
    


        self.g_a_progressive = nn.Sequential(
            conv(self.T, N, kernel_size=5, stride=2), # halve 2 so 128
            GDN(N),
            conv(N, N, kernel_size=5, stride=2), # halve 4 so 64 k**2 = 16  
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
            conv(N, N, kernel_size=5, stride=2), #halve 8 so dim is 32  k**2 = 64 
            GDN(N),
            conv(N, M, kernel_size=5, stride=2), # 16 
        )


        if self.mask_policy == "learnable-mask":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels - 1, self.M))) #il primo e il base layer
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M, out_channels=self.M, kernel_size=1, stride=1),)



        self.h_a_prog = nn.Sequential(
            conv3x3(self.M, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, self.N, stride=2),
        )
  

        self.h_mean_prog = nn.Sequential(
            conv3x3(self.N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, self.M),
        )

        self.h_scale_prog = nn.Sequential(
            conv3x3(self.N, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, self.M),
        )



    def split_ga(self, x, begin = True):
        if begin:
            layers_intermedi = list(self.g_a.children())[:self.level + 1]
        else:
            layers_intermedi = list(self.g_a.children())[self.level + 1:]
        modello_intermedio = nn.Sequential(*layers_intermedi)
        return modello_intermedio(x)


    def concatenate(self, y_base, x):
        bs,c,w,h = y_base.shape 
        y_base = y_base.reshape(bs,c//self.factor, w*self.halve, h*self.halve).to(x.device)
        res = torch.cat([y_base,x],dim = 1).to(x.device)
        return res 

    def extract_mask(self,scale,  pr = 0):

        shapes = scale.shape
        bs, ch, w,h = shapes
        if self.mask_policy == "point-based-std":
            assert scale is not None 
            pr = pr*0.1  
            scale = scale.ravel()
            quantile = torch.quantile(scale, pr)
            res = scale >= quantile 
            #print("dovrebbero essere soli 1: ",torch.unique(res, return_counts = True))
            return res.reshape(bs,ch,w,h).to(torch.float)
        elif self.mask_policy == "learnable-mask":
            if self.lmbda_index_list[pr] == 0:
                return torch.zeros_like(scale).to(scale.device)
            
            importance_map =  self.mask_conv(scale) 
            importance_map = torch.clip(importance_map + 0.5, 0, 1)
            gamma = torch.sum(torch.stack([self.gamma[j] for j in range(self.lmbda_index_list[pr])]),dim = 0) # più uno l'hom esso in lmbda_index
            gamma = gamma[None, :, None, None]
            gamma = torch.relu(gamma)

            
            adjusted_importance_map = torch.pow(importance_map, gamma)
            return adjusted_importance_map          
        else:
            raise NotImplementedError()
        
        
    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality = None):

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]

        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)


        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # calcoliamo mean and std per il progressive
        
        z_prog = self.h_a_prog(y_progressive)
        _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog)

        z_offset_prog = self.entropy_bottleneck_prog._get_medians()
        z_tmp_prog = z_prog - z_offset_prog
        z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

        scales_prog = self.h_scale_prog(z_hat_prog)
        means_prog = self.h_mean_prog(z_hat_prog)

        list_quality = self.lmbda_list if quality is None else [quality]


        y_slices = y.chunk(self.num_slices, 1)


        y_likelihoods_progressive = []
        y_likelihood_main = []

        x_hat_progressive = []
        



        for j,p in enumerate(list_quality): 
            mask = self.extract_mask(latent_scales,quality = p)
            if self.mask_policy == "learnable-mask" and self.lmbda_index_list[p]!=0:
                # Create a relaxed mask for quantization
                samples = mask + (torch.rand_like(mask) - 0.5)
                mask = samples + samples.round().detach() - samples.detach()  # Differentiable torch.round()           
            
            y_prog_q_zero = y_progressive - means_prog 
            y_prog_q= mask*y_prog_q_zero 


            _,y_prog_q_likelihood = self.gaussian_conditional(y_prog_q, scales_prog*mask) 

            y_likelihoods_progressive.append(y_prog_q_likelihood.unsqueeze(0))

            y_prog_q = ste_round(y_prog_q) + means_prog 


            y_prog_q_slices = y_prog_q.chunck(self.num_slices,dim = 1)
            y_hat_slices = []
            y_hat_slices_scalable = []
            

            for slice_index, y_slice in enumerate(y_slices):
                y_hat_prog_slice = y_prog_q_slices[slice_index]
                support_index = min(max(0,slice_index - self.max_support_slices),self.max_support_slices)
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[support_index:])
                
                # encode the main latent representation 
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                
                
                if j == 0:
                    y_likelihood_main.append(y_slice_likelihood)
                y_hat_slice_base = ste_round(y_slice - mu) + mu  

                y_hat_slice = y_hat_slice_base + y_hat_prog_slice 

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp


                y_hat_slices.append(y_hat_slice_base)
                y_hat_slices_scalable.append(y_hat_slice)

            y_hat_q = torch.cat(y_hat_slices_scalable,dim = 1) #questo va preso 

            x_hat_q = self.g_s(y_hat_q) 

            x_hat_progressive.append(x_hat_q.unsqueeze(0))       

        x_hat_progressive = torch.cat(x_hat_progressive,dim = 0) #num_scalable, BS,3,W,H 
        y_likelihoods = torch.cat(y_likelihood_main, dim = 0) # BS,3,W,H solo per base 
        y_likelihoods_prog = torch.cat(y_likelihoods_progressive,dim = 0)
         

        return {
            "x_hat": x_hat_progressive,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_prog},

        }
    

    def compress(self, x, p = 0.0):
        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        
        y_residual_input = self.concatenate(y_base,x)
        y_residual = self.g_a_residual(y_residual_input)

        y_residual = self.masking(y_residual, p = p)
        y = y + y_residual # maschero per ora qua, poi ci pensiamo

        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}



class ResWACNN2018(ResWACNN):
    "evolution of the previous one, but with oldest entropy estimation"

    def __init__(self, N=192, M=320,mask_policy = "block-wise", scalable_levels = 4, **kwargs):
        super().__init__(N = N, M = M,mask_policy = mask_policy, **kwargs)


        #definire 
        self.scalable_levels = scalable_levels  
        self.percentages = [0,5,10]#list(range(0,10,10//self.scalable_levels))
        print(self.percentages,"<------")
        self.entropy_bottleneck_residual = EntropyBottleneck(self.N)

        self.h_a_res = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N),
        )

        self.h_s_res = nn.Sequential(
            deconv(self.N, self.M),
            nn.LeakyReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )


        if self.mask_policy == "learnable-mask":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels, 320)))
            self.gamma[0,:].data *= 0.0
            
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M, out_channels=self.M, kernel_size=1, stride=1),)
           




    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_residual: ",sum(p.numel() for p in self.g_a_residual.parameters()))
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print(" h_a_res: ",sum(p.numel() for p in self.h_a_res.parameters()))
        print(" h_s_res: ",sum(p.numel() for p in self.h_s_res.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.lrp_transforms.parameters()))

        if self.mask_policy== "learnable-mask":
            print("mask conv",sum(p.numel() for p in self.mask_conv.parameters()))
           

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)





    def forward(self, x):

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)


        #residual latent representation 
        y_residual_input = self.concatenate(y_base,x).to(y_base.device)
        y_residual = self.g_a_residual(y_residual_input)

        y_residual = y_residual.to(x.device)

 

        y_shape = y.shape[2:]

    
        # hyperprior for main layer 
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        #overall latent scales and means for the main latent representation
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # hyperprior for residual 
        z_res =self.h_a_res(y_residual)
        _, z_likelihoods_res = self.entropy_bottleneck_residual(z_res) 

        z_offset_res = self.entropy_bottleneck_residual._get_medians()
        z_tmp = z_res - z_offset_res
        z_hat_res = ste_round(z_tmp) + z_offset_res 

        

        gaussian_params = self.h_s_res(z_hat_res)
        scales_hat_res, means_hat_res = gaussian_params.chunk(2, 1)
        #scales_hat_res = scales_hat_res.to(y.device)
        #means_hat_res = scales_hat_res.to(y.device)



        y_hat_residual_across_percentages = []
        y_likelihoods_res_across_percentages = []

        #for p in self.percentages:
            #print("*********** ",p," *******************")
        #    mask = self.masking(y.shape,p = p)  # determina la maschera
            #print("mask shape",mask.device)
            #print("y_residual: ",y_residual.device)
            #print("y: ",y.shape)
        #    y_hat_residual, y_likelihoods_residual = self.gaussian_conditional(y_residual*mask, 
                                                                           #     scales_hat_res*mask,
                                                                           #     means= means_hat_res*mask) 
        #    y_hat_residual_across_percentages.append(y_hat_residual) 
        #    y_likelihoods_res_across_percentages.append(y_likelihoods_residual)

        

        y_slices = y.chunk(self.num_slices, 1)
        
        y_hat_scalable = []
        y_likelihood_scalable = []

        for i in range(len((self.percentages))):


            mask = self.masking(scales_hat_res,p = i)  # determina la maschera


            if self.mask_policy == "learnable-mask":
                if self.training:
                    mask = mask + (torch.rand_like(mask) - 0.5)
                    mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()
                else:
                    mask = torch.round(mask)



            y_hat_residual, y_likelihoods_residual = self.gaussian_conditional(y_residual*mask, 
                                                                                scales_hat_res*mask,
                                                                                means= means_hat_res*mask) 


            y_hat_residual_across_percentages.append(y_hat_residual) 
            y_likelihoods_res_across_percentages.append(y_likelihoods_residual)

            y_residual_p = y_hat_residual_across_percentages[i]
            y_residual_slices = y_residual_p.chunk(self.num_slices,1)

            y_hat_slices_p = []
            y_likelihood_p = []

            for slice_index, y_slice in enumerate(y_slices):


                y_slice_residual = y_residual_slices[slice_index]

                support_slices = (y_hat_slices_p if self.max_support_slices < 0 else y_hat_slices_p[:self.max_support_slices])



                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                y_likelihood_p.append(y_slice_likelihood)


                #quantize the entire 
                y_hat_slice_zero_residual= ste_round(y_slice - mu) + mu
                
                y_hat_slice = y_hat_slice_zero_residual + y_slice_residual # add the residual!


                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp

                y_hat_slices_p.append(y_hat_slice)
            
            y_hat_p = torch.cat(y_hat_slices_p,dim = 1) 
            y_likelihood_p = torch.cat(y_likelihood_p,dim = 1) 

            y_hat_scalable.append(y_hat_p)
            y_likelihood_scalable.append(y_likelihood_p)


        x_hat_scalable = []
        for i,p in enumerate(self.percentages):
            y_hat = y_hat_scalable[i]
            x_hat = self.g_s(y_hat)
            x_hat_scalable.append(x_hat) #serve per la dimensionalità

        
        x_hat_scalable = torch.cat(x_hat_scalable, dim = 0) #bs*percentage,3,w,h
        y_likelihood_scalable = torch.cat(y_likelihood_scalable,dim =0) #bs*percentage,320,16,16

        y_likelihoods_res_across_percentages = torch.cat(y_likelihoods_res_across_percentages,dim = 0)


        


        return {
            "x_hat": x_hat_scalable,
            "likelihoods": {"y": y_likelihood_scalable,
                             "r":y_likelihoods_res_across_percentages},
            "likelihoods_hyperprior":{"z": z_likelihoods,
                                       "z_res":z_likelihoods_res},
            "latent": {"y_hat":y_hat_scalable,
                       "res":y_hat_residual}
        }
    


    def forward_test(self,x,scale_p):

        assert scale_p < len(self.percentages)

        #p = self.percentages[scale_p]

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)

        #residual latent representation 
        y_residual_input = self.concatenate(y_base,x)
        y_residual = self.g_a_residual(y_residual_input)


        y_shape = y.shape[2:]

    
        # hyperprior for main layer 
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        #overall latent scales and means for the main latent representation
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # hyperprior for residual 
        z_res =self.h_a_res(y_residual)
        _, z_likelihoods_res = self.entropy_bottleneck_residual(z_res) 

        z_offset_res = self.entropy_bottleneck_residual._get_medians()
        z_tmp = z_res - z_offset_res
        z_hat_res = ste_round(z_tmp) + z_offset_res 

        

        gaussian_params = self.h_s_res(z_hat_res)
        scales_hat_res, means_hat_res = gaussian_params.chunk(2, 1)


        #y_hat_residual_across_percentages = []
        #y_likelihoods_res_across_percentages = []

        
        mask = self.masking(scales_hat_res,p = scale_p)  # determina la maschera

        if self.mask_policy == "learnable-mask":
            mask = torch.round(mask)
        y_hat_residual, y_likelihoods_residual = self.gaussian_conditional(y_residual*mask, 
                                                                                scales_hat_res*mask,
                                                                                means= means_hat_res*mask) 

        

        y_slices = y.chunk(self.num_slices, 1)
        y_residual_slices = y_hat_residual.chunk(self.num_slices,1)        
        y_hat_slices = []
        y_likelihood_slices = []



        for slice_index, y_slice in enumerate(y_slices):


            y_slice_residual = y_residual_slices[slice_index]

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])



            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood_slices.append(y_slice_likelihood)


            #quantize the entire 
            y_hat_slice_zero_residual= ste_round(y_slice - mu) + mu
                
            y_hat_slice = y_hat_slice_zero_residual + y_slice_residual # add the residual!


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            
        y_hat = torch.cat(y_hat_slices,dim = 1) 
        y_likelihood = torch.cat(y_likelihood_slices,dim = 1) 


        x_hat = self.g_s(y_hat)




        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihood,
                             "r":y_likelihoods_residual},
            "likelihoods_hyperprior":{"z": z_likelihoods,
                                       "z_res":z_likelihoods_res},
            "latent": {"y_hat":y_hat,
                       "res":y_hat_residual}
        }
           


    def compress(self, x, p = 0):


        assert p < len(self.percentages) 

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)

        #residual latent representation 
        y_residual_input = self.concatenate(y_base,x)
        y_residual = self.g_a_residual(y_residual_input)


        y_shape = y.shape[2:]


        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        # turn of the residual
        z_res = self.h_a_res(y_residual)
        z_string_res = self.entropy_bottleneck_residual.compress(z_res)
        z_hat_res = self.entropy_bottleneck.decompress(z_string_res , z_res.size()[-2:])

        gaussian_params = self.h_s_res(z_hat_res)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        mask = self.masking(scales_hat, p = p)  # determina la maschera

        if self.masking == "learnable-mask":
            mask = torch.round(mask)
        

        indexes = self.gaussian_conditional.build_indexes(scales_hat*mask)
        y_strings_res = self.gaussian_conditional.compress(y_residual*mask, indexes, means=means_hat*mask)

        y_hat_residual = self.gaussian_conditional.decompress(y_strings_res, indexes, means=means_hat*mask)



        y_slices = y.chunk(self.num_slices, 1)
        y_res_slice = y_hat_residual.chunck(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)


            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            y_hat_slice_zero = y_q_slice + mu
            y_hat_slice = y_hat_slice_zero + y_res_slice

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings, y_strings_res,z_string_res ], "shape": [z.size()[-2:], z_res.size()[-2:]]}

    



    def decompress(self, strings, shape,p):
        assert p < len(self.percentages)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        z_hat_res = self.entropy_bottleneck_residual.decompress(strings[-1],shape[-1]) 
        gaussian_params = self.h_s_res(z_hat_res)
        scales_hat, means_hat = gaussian_params.chunk(2, 1) 

        mask = self.masking(scales_hat,p = p)

        if self.masking == "learnable-mask":
            mask = torch.round(mask)

        indexes = self.gaussian_conditional.build_indexes(scales_hat*mask)
        y_hat_residual = self.gaussian_conditional.decompress(strings[2], indexes, means=means_hat)



        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        y_hat_residual_slices = y_hat_residual.chuck(self.num_slices,1)

        for slice_index in range(self.num_slices):
            y_residual_slice = y_hat_residual_slices[slice_index]
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            y_hat_slice = y_hat_slice + y_residual_slice

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}