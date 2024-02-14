
import math
import torch
import torch.nn as nn
from compress.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compress.entropy_models import GaussianConditionalMask
from ..utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3
from compress.layers.mask_layer import Mask
from compressai.ans import BufferedRansEncoder, RansDecoder

from ..cnn import WACNN
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64




def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ConditionalSharedWACNN(WACNN):
    """CNN based model"""

    def __init__(self, 
                 N=192,
                M=320,
                mask_policy = "scalable_res",
                joiner_policy = "conditional",
                lmbda_list = None,
                independent_hyperprior = False,
                **kwargs):
        super().__init__(N = N, M = M, **kwargs)

        self.independent_hyperprior = independent_hyperprior
        assert joiner_policy in ("conditional","residual","concatenation","cac")
        self.joiner_policy = joiner_policy
        self.N = N 
        self.M = M 
        self.halve = 8
        self.level = 5 if self.halve == 8 else -1 
        self.factor = self.halve**2
        assert self.N%self.factor == 0 
        self.T = int(self.N//self.factor) + 3
        self.mask_policy = mask_policy


        self.scalable_levels = len(lmbda_list)
        self.lmbda_list = lmbda_list
        self.lmbda_index_list = dict(zip(self.lmbda_list, [i  for i in range(len(self.lmbda_list))] ))

        self.dimensions_M = [self.M, self.M*2 if self.joiner_policy == "concatenation" else self.M]

        
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_prog = GaussianConditionalMask(None)
    
        self.masking = Mask(self.mask_policy,self.scalable_levels,self.M )
        """
        if self.mask_policy == "learnable-mask-gamma":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels - 1, self.M))) #il primo e il base layer, lìultimo è il completo!!!
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M*2, out_channels=self.M, kernel_size=1, stride=1),)
        if self.mask_policy == "learnable-mask-nested":
            self.mask_conv = nn.ModuleList(
                            nn.Sequential(torch.nn.Conv2d(in_channels=self.M*2, 
                                                          out_channels=self.M, 
                                                          kernel_size=1, 
                                                          stride=1),)
                            for _ in range(len(self.lmbda_list) -1)
            
                            )

        """
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



        if self.independent_hyperprior:

            self.entropy_bottleneck_prog = EntropyBottleneck(self.N)
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

        if self.joiner_policy == "conditional":
            self.joiner = nn.ModuleList(
                nn.Sequential(
                    conv(64, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )
        elif self.joiner_policy == "cac":
            self.joiner_policy = nn.Conv2d(in_channels=M,
                                                       out_channels=M,
                                                         kernel_size=1, 
                                                         stride=1, 
                                                         padding=0)
                
        self.g_s = nn.ModuleList(
                    nn.Sequential(
                    Win_noShift_Attention(dim= self.dimensions_M[i], num_heads=8, window_size=4, shift_size=2),
                    deconv(self.dimensions_M[i], N, kernel_size=5, stride=2),
                    GDN(N, inverse=True),
                    deconv(N, N, kernel_size=5, stride=2),
                    GDN(N, inverse=True),
                    Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                    deconv(N, N, kernel_size=5, stride=2),
                    GDN(N, inverse=True),
                    deconv(N, 3, kernel_size=5, stride=2),
            ) for i in range(2) 
        )


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = self.gaussian_conditional_prog.update_scale_table(scale_table, force=force)
        if self.independent_hyperprior:
            self.entropy_bottleneck_prog.update()

        self.entropy_bottleneck.update()
        return updated

    def extract_mu_and_scale(self,mean_support, scale_support,slice_index,y_shape ):
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]] 
        return mu, scale 


    def load_state_dict(self, state_dict):


        update_registered_buffers(
            self.gaussian_conditional_prog,
            "gaussian_conditional_prog",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )

        if self.independent_hyperprior:
            update_registered_buffers(
                self.entropy_bottleneck_prog,
                "entropy_bottleneck_prog",
                ["_quantized_cdf", "_offset", "_cdf_length"],
                state_dict,
            )
            super().load_state_dict(state_dict)


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
    

    def clip_with_relu(self,x,min_val, max_val):
        relu_min = torch.relu(x - min_val)
        relu_max = torch.relu(max_val - x)
        result = x - relu_min - relu_max
        return result     


    """
    def extract_mask(self,scale,scale_prog = None,  pr = 0):

        shapes = scale.shape
        bs, ch, w,h = shapes
        if self.mask_policy == "point-based-std":
            if pr == 1.0:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0.0:
                return torch.zeros_like(scale).to(scale.device)
            
            assert scale is not None 
            pr = 1.0 - pr
            scale = scale.ravel()
            quantile = torch.quantile(scale, pr)
            res = scale >= quantile 
            #print("dovrebbero essere soli 1: ",torch.unique(res, return_counts = True))
            return res.reshape(bs,ch,w,h).to(torch.float)
        elif self.mask_policy == "learnable-mask-gamma":
            
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)

            assert scale_prog is not None 
            scale_input = torch.cat([scale,scale_prog],dim = 1)
            importance_map =  self.mask_conv(scale_input) 

            importance_map = torch.sigmoid(importance_map + 0.5) 

            idx = pr - 1
            gamma = self.gamma[idx][None, :, None, None]
            gamma = torch.relu(gamma) 


            adjusted_importance_map = torch.pow(importance_map, gamma)
            #adjusted_importance_map.register_hook(lambda grad: print('adjusted_importance_map back', (grad != grad).any().item()))
            return adjusted_importance_map          

        elif self.mask_policy == "learnable-mask-nested":

            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            
            assert scale_prog is not None 
            scale_input = torch.cat([scale,scale_prog],dim = 1)

            importance_map = torch.sum(torch.stack([torch.sigmoid(self.mask_conv[i](scale_input)) for i in range(pr)],dim = 0),dim = 0) 
            importance_map = torch.sigmoid(importance_map)    

            return importance_map       

        elif self.mask_policy == "two-levels":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            else:
                return torch.ones_like(scale).to(scale.device)
        
        elif self.mask_policy == "scalable_res":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            elif pr == len(self.lmbda_list) - 1:
                return torch.ones_like(scale).to(scale.device)
            else: 
                c = torch.zeros_like(scale).to(scale.device)
                lv = self.M - 32*pr*2 
                c[:,lv:,:,:] = 1.0

                return c.to(scale.device)     
        else:
            raise NotImplementedError()
    """ 

    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))


        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        if "learnable-mask" in self.mask_policy:
            print("mask conv",sum(p.numel() for p in self.masking.mask_conv.parameters()))
            if "gamma" is self.mask_policy:
                print("gamma",sum(p.numel() for p in self.masking.gamma.parameters()))

        
        if self.independent_hyperprior:
            print(" h_mean_prog: ",sum(p.numel() for p in self.h_mean_prog.parameters()))
            print(" h_scale_prog: ",sum(p.numel() for p in self.h_scale_prog.parameters()))
            print(" h_a_prog: ",sum(p.numel() for p in self.h_a_prog.parameters()))
            print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))

        for i in range(2):
            print(" g_s_" + str(i) + ":" ,sum(p.numel() for p in self.g_s[i].parameters()))

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



    def determ_quality(self,quality):
        if quality is None:
            list_quality = self.lmbda_list 
        elif isinstance(quality,list):
            list_quality = quality 
        else:
            list_quality = [quality] 
        return list_quality
    



    def merge(self,y_main,y_prog, slice = 0):
        if self.joiner_policy == "residual":
            return y_main + y_prog 
        elif self.joiner_policy == "concatenation" or self.joiner_policy == "cac":
            return y_main #torch.cat([y_main, y_prog], dim=1).to(y_main.device)
        elif self.joiner_policy == "block_concatenation":
            return torch.cat([y_main, y_prog], dim=1).to(y_main.device)


        else:
            y_hat_slice_support = torch.cat([y_main, y_prog], dim=1)
            return self.joiner[slice](y_hat_slice_support)


    def hyperEncoderDecoder(self,y, progressive = False):
        if progressive:
            z_prog = self.h_a_prog(y)
            _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog)

            z_offset_prog = self.entropy_bottleneck_prog._get_medians()
            z_tmp_prog = z_prog - z_offset_prog
            z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

            scales_prog = self.h_scale_prog(z_hat_prog)
            means_prog = self.h_mean_prog(z_hat_prog)
            return z_likelihoods_prog, scales_prog, means_prog
        else:
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            scales = self.h_scale_s(z_hat)
            means = self.h_mean_s(z_hat)
            return z_likelihoods, scales, means


    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality = 0):

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]

        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)


        z_likelihoods, latent_scales, latent_means = self.hyperEncoderDecoder(y)

        
        z_likelihoods_prog, scales_prog, means_prog = self.hyperEncoderDecoder(y_progressive,
                                                        True if self.independent_hyperprior \
                                                        else False)


        y_slices = y.chunk(self.num_slices, 1)
        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)

        y_likelihood_main = []
        y_likelihood_progressive = []
        y_hat_slices = []
        y_hat_slices_prog = []

        mask = self.masking(latent_scales,pr = quality)
        if "learnable-mask" in self.mask_policy: # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
            mask = self.masking.apply_noise(mask,self.training)
        mask_slices = mask.chunk(self.num_slices,dim = 1)
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            y_progressive_slice = y_progressive_slices[slice_index]
            block_mask = mask_slices[slice_index]

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood_main.append(y_slice_likelihood)

            y_hat_slice_main = ste_round(y_slice - mu) + mu 

            lrp_support = torch.cat([mean_support, y_hat_slice_main], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice_main += lrp


            mean_support_prog = torch.cat([means_prog] + support_slices, dim=1)
            scale_support_prog = torch.cat([scales_prog] + support_slices, dim=1)
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, scale_support_prog, slice_index, y_shape)

            _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_progressive_slice, 
                                                                        scale_prog*block_mask,
                                                                        mu_prog,
                                                                        mask = block_mask)
   
            y_likelihood_progressive.append(y_slice_likelihood_prog)

            y_hat_slice_progressive = ste_round(y_progressive_slice - mu_prog)*block_mask + mu_prog

            lrp_support_prog = torch.cat([mean_support_prog, y_hat_slice_progressive], dim=1)
            lrp_prog = self.lrp_transforms[slice_index](lrp_support_prog)
            lrp_prog = 0.5 * torch.tanh(lrp_prog)
            y_hat_slice_progressive += lrp_prog

            y_hat_slice = self.merge(y_hat_slice_main, y_hat_slice_progressive, slice_index)
            y_hat_slices_prog.append(y_hat_slice_progressive)

            y_hat_slices.append(y_hat_slice)



        if self.joiner_policy == "concatenation":

            y_hat = torch.cat(y_hat_slices +  y_hat_slices_prog , dim=1) if quality == 1 else torch.cat(y_hat_slices,dim = 1)
        elif self.joiner_policy == "cac":
            y_hat = torch.cat(y_hat_slices +  y_hat_slices_prog , dim=1)
            y_hat = self.joiner(y_hat)
        else:
            y_hat = torch.cat(y_hat_slices, dim=1)


        y_likelihoods_main = torch.cat(y_likelihood_main, dim=1)
        y_likelihoods_progressive = torch.cat(y_likelihood_progressive)
        x_hat = self.g_s[0 if quality == 0 else 1](y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods_main, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_progressive},
        }
    




    def compress(self, x, quality = 0.0, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        y_base= self.split_ga(x)
        y = self.split_ga(y_base,begin = False)
        y_shape = y.shape[2:]


        y_progressive_support = self.concatenate(y_base,x)
        y_progressive = self.g_a_progressive(y_progressive_support)

        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()

        cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        encoder_prog = BufferedRansEncoder()


        symbols_list = []
        indexes_list = []

        symbols_list_prog = []
        indexes_list_prog = []

        y_strings = []
        y_strings_prog = []


        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality



        if self.independent_hyperprior:
            z_prog = self.h_a_prog(y_progressive)
            z_string_prog = self.entropy_bottleneck_prog.compress(z_prog)
            z_hat_prog = self.entropy_bottleneck_prog.decompress(z_string_prog,z_prog.size()[-2:])
            latent_scales_prog = self.h_scale_prog(z_hat_prog)
            latent_means_prog = self.h_mean_prog(z_hat_prog)
        else:
            z_prog = self.h_a(y_progressive)
            z_string_prog = self.entropy_bottleneck.compress(z_prog)
            z_hat_prog = self.entropy_bottleneck.decompress(z_string_prog,z_prog.size()[-2:])
            latent_scales_prog = self.h_scale_s(z_hat_prog)
            latent_means_prog = self.h_mean_s(z_hat_prog)

        mask = self.masking(scale = latent_scales_prog, pr =q, mask_pol = mask_pol)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)
        

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_hat_slices_prog = []

        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)

        for slice_index, y_slice in enumerate(y_slices):
            #part con la parte main 
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice_main = y_q_slice + mu
       
            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice_main], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice_main += lrp

            #prendo i progressive
            block_mask = mask_slices[slice_index]
            y_slice_prog = y_progressive_slices[slice_index]

            mean_support_prog = torch.cat([latent_means_prog] + support_slices, dim=1)
            scale_support_prog = torch.cat([latent_scales_prog] + support_slices, dim=1)
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, scale_support_prog, slice_index, y_shape)

            index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
            index_prog = index_prog.int()

            y_q_slice_prog = self.gaussian_conditional_prog.quantize(y_slice_prog, "symbols",mu_prog, mask=block_mask)
            y_hat_slice_prog = y_q_slice_prog + mu_prog


            symbols_list_prog.extend(y_q_slice_prog.reshape(-1).tolist())
            indexes_list_prog.extend(index_prog.reshape(-1).tolist())


            lrp_support_prog = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
            lrp_prog = self.lrp_transforms[slice_index](lrp_support_prog)
            lrp_prog = 0.5 * torch.tanh(lrp_prog)
            y_hat_slice_prog += lrp_prog

            y_hat_slice = self.merge(y_hat_slice_main, y_hat_slice_prog, slice_index)
            y_hat_slices_prog.append(y_hat_slice_prog)

            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        encoder_prog.encode_with_indexes(symbols_list_prog, indexes_list_prog, cdf_prog, cdf_lengths_prog, offsets_prog)

        y_string_prog = encoder_prog.flush()
        y_strings_prog.append(y_string_prog)



        
        return {"strings": [y_strings, z_strings, z_string_prog,y_strings_prog],  #preogressive_strings
                "shape": [z.size()[-2:],z_prog.size()[-2:]],          
                }   
    


    def decompress(self, strings, shape, quality, mask_pol = None):

        if mask_pol is None:
            mask_pol = self.mask_policy

        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_string_prog = strings[-1][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)


        cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        decoder_prog = RansDecoder()
        decoder_prog.set_stream(y_string_prog)

        

        if self.independent_hyperprior:
            z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])
            latent_scales_prog = self.h_scale_prog(z_hat_prog)
            latent_means_prog = self.h_mean_prog(z_hat_prog)
        else:
            z_hat_prog =  self.entropy_bottleneck.decompress(strings[2],shape[-1])
            latent_scales_prog = self.h_scale_s(z_hat_prog)
            latent_means_prog = self.h_mean_s(z_hat_prog)


        mask = self.masking(scale = latent_scales_prog, pr =q, mask_pol = mask_pol)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)

        y_hat_slices = []
        y_hat_slices_prog = []


        for slice_index in range(self.num_slices):
            block_mask = mask_slices[slice_index]

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale =  self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)
            
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice_main = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice_main], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice_main += lrp

           
            mean_support_prog = torch.cat([latent_means_prog] + support_slices, dim=1)
            scale_support_prog = torch.cat([latent_scales_prog] + support_slices, dim=1)
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, scale_support_prog, slice_index, y_shape)

            index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
            index_prog = index_prog.int()      


            rv_prog = decoder_prog.decode_stream(index_prog.reshape(-1).tolist(),
                                                cdf_prog, 
                                               cdf_lengths_prog, 
                                                offsets_prog)
            rv_prog = torch.Tensor(rv_prog).reshape(mu_prog.shape)


            y_hat_slice_prog = self.gaussian_conditional_prog.dequantize(rv_prog, means = mu_prog)

            lrp_support_prog = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
            lrp_prog = self.lrp_transforms[slice_index](lrp_support_prog)
            lrp_prog = 0.5 * torch.tanh(lrp_prog)
            y_hat_slice_prog += lrp_prog

            y_hat_slice = self.merge(y_hat_slice_main, y_hat_slice_prog, slice_index)
            y_hat_slices_prog.append(y_hat_slice_prog)

            y_hat_slices.append(y_hat_slice)


        if self.joiner_policy == "concatenation":
            y_hat = torch.cat(y_hat_slices +  y_hat_slices_prog , dim=1) if quality != 0 else torch.cat(y_hat_slices,dim = 1)
        elif self.joiner_policy == "cac":
            y_hat = torch.cat(y_hat_slices +  y_hat_slices_prog , dim=1)
            y_hat = self.joiner(y_hat)
        else:
            y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s[0 if quality == 0 else 1](y_hat).clamp_(0,1)
        return {"x_hat": x_hat}  