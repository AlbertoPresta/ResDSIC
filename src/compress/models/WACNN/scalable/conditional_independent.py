import math
import torch
import torch.nn as nn
from compress.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional
from ..utils import conv, deconv, update_registered_buffers
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3


from ..cnn import WACNN
from compress.layers import conv3x3, subpel_conv3x3, Win_noShift_Attention
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64




def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class ConditionalWACNN(WACNN):
    """CNN based model"""

    def __init__(self, 
                 N=192,
                M=320,
                mask_policy = "learnable-mask",
                joiner_policy = "conditional",
                lmbda_list = None,
                **kwargs):
        super().__init__(N = N, M = M, **kwargs)


        assert joiner_policy in ("conditional","residual","concatenation")

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


        self.entropy_bottleneck_prog = EntropyBottleneck(self.N)
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_prog = GaussianConditional(None)
    

        if self.mask_policy == "learnable-mask":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels - 1, self.M))) #il primo e il base layer
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M, out_channels=self.M, kernel_size=1, stride=1),)




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

        if self.joiner_policy == "concatenation": #ssss

            self.g_s = nn.Sequential(
                Win_noShift_Attention(dim=M*2, num_heads=8, window_size=4, shift_size=2),
                deconv(M*2, M, kernel_size=5, stride=2),
                GDN(M, inverse=True),
                deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
            )

            """
            self.lrp_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 64 * min(i+1, 6), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )


            self.cc_mean_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 64*min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
            )
            self.cc_scale_transforms = nn.ModuleList(
                nn.Sequential(
                    conv(320 + 64 * min(i, 5), 224, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(224, 176, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(176, 128, stride=1, kernel_size=3),
                    nn.GELU(),
                    conv(128, 64, stride=1, kernel_size=3), #sss
                    nn.GELU(),
                    conv(64, 32, stride=1, kernel_size=3),
                ) for i in range(10)
                )
            """


        self.cc_mean_transforms_prog = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
        )
        self.cc_scale_transforms_prog = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            ) for i in range(10)
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


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss
    


    def load_state_dict(self, state_dict):


        update_registered_buffers(
            self.gaussian_conditional_prog,
            "gaussian_conditional_prog",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )


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



    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = self.gaussian_conditional_prog.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)

        self.entropy_bottleneck_prog.update(force=force)
        return updated


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
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if pr == len(self.lmbda_list) -1:
                return torch.ones_like(scale).to(scale.device)
  
            importance_map =  self.mask_conv(scale) 
            importance_map = self.clip_with_relu(importance_map + 0.5, 0,1).to(scale.devices) #torch.clip(importance_map + 0.5, 0, 1)
            gamma = torch.sum(torch.stack([self.gamma[j] for j in range(pr)]),dim = 0) # più uno l'hom esso in lmbda_index
            gamma = gamma[None, :, None, None]
            gamma = torch.relu(gamma)

            
            adjusted_importance_map = torch.pow(importance_map, gamma)
            return adjusted_importance_map          
        elif self.mask_policy == "two-levels":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            else:
                return torch.ones_like(scale).to(scale.device)
        else:
            raise NotImplementedError()
        


    def determ_quality(self,quality):
        if quality is None:
            list_quality = self.lmbda_list 
        elif isinstance(quality,list):
            list_quality = quality 
        else:
            list_quality = [quality] 
        return list_quality
    


    def extract_mu_and_scale(self,mean_support, scale_support,slice_index,y_shape, prog = False ):

            if prog is False:
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]] 

                return mu, scale 

            else:
                mu = self.cc_mean_transforms_prog[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms_prog[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]] 

                return mu, scale 



    def merge(self,y_main,y_prog, slice = 0):
        if self.joiner_policy == "residual":
            return y_main + y_prog 
        elif self.joiner_policy == "concatenation":
            return y_main #torch.cat([y_main, y_prog], dim=1).to(y_main.device)
        else:
            y_hat_slice_support = torch.cat([y_main, y_prog], dim=1)
            return self.joiner[slice](y_hat_slice_support)

    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality = 0):

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
        
        z_prog = self.h_a_prog(y_progressive)
        _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog)

        z_offset_prog = self.entropy_bottleneck_prog._get_medians()
        z_tmp_prog = z_prog - z_offset_prog
        z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

        scales_prog = self.h_scale_prog(z_hat_prog)
        means_prog = self.h_mean_prog(z_hat_prog)

        y_slices = y.chunk(self.num_slices, 1)
        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)

        y_likelihood_main = []
        y_likelihood_progressive = []
        y_hat = []
        y_hat_slices = []

        mask = self.extract_mask(latent_scales,pr = quality)
        if self.mask_policy == "learnable-mask": # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
            if self.training:
                mask = mask + (torch.rand_like(mask) - 0.5)
                mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()   
            else:
                mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)
        y_hat_slices_prog = []
        for slice_index, y_slice in enumerate(y_slices):

            y_progressive_slice = y_progressive_slices[slice_index]
            block_mask = mask_slices[slice_index]

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            # encode the main latent representation 

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
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, scale_support_prog, slice_index, y_shape, prog = True)

            _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_progressive_slice, scale_prog,mu_prog, mask = block_mask)
   
            y_likelihood_progressive.append(y_slice_likelihood_prog)

            y_hat_slice_progressive = ste_round(y_progressive_slice - mu_prog)*block_mask + mu_prog

            y_hat_slice = self.merge(y_hat_slice_main, y_hat_slice_progressive, slice_index)
            y_hat_slices_prog.append(y_hat_slice_progressive)

            y_hat_slices.append(y_hat_slice)



        if self.joiner_policy == "concatenation":

            y_hat = torch.cat(y_hat_slices +  y_hat_slices_prog , dim=1)
        else:
            y_hat = torch.cat(y_hat_slices, dim=1)


        y_likelihoods_main = torch.cat(y_likelihood_main, dim=1)
        y_likelihoods_progressive = torch.cat(y_likelihood_progressive)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods_main, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_progressive},
        }

        

    def compress(self, x, quality = 0.0):
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

        #cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        #cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        #offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        #encoder_prog = BufferedRansEncoder()

        symbols_list = []
        indexes_list = []

        #symbols_list_prog = []
        #indexes_list_prog = []

        y_strings = []
        y_strings_prog = []


        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality


        #if q != 0:self.def
        z_prog = self.h_a_prog(y_progressive)
        z_string_prog = self.entropy_bottleneck_prog.compress(z_prog)
        z_hat_prog = self.entropy_bottleneck_prog.decompress(z_string_prog,z_prog.size()[-2:])
        latent_scales_prog = self.h_scale_prog(z_hat_prog)
        latent_means_prog = self.h_mean_prog(z_hat_prog)

        mask = self.extract_mask(scale = latent_scales_prog, pr =q)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)

        #progressive_strings = []

    
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)

        progressive_strings = []
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

            #[latent_means] + support_slices
            mean_support_prog = torch.cat([latent_means_prog] + support_slices, dim=1)
            scale_support_prog = torch.cat([latent_scales_prog] + support_slices, dim=1)
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog,scale_support_prog,slice_index,y_shape,prog = True)

            #index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
            #index_prog = index_prog.int()

            #y_q_slice_prog = self.gaussian_conditional_prog.quantize(y_slice_prog, "symbols",mu_prog)
            #y_hat_slice_prog = y_q_slice_prog + mu_prog

            #symbols_list_prog.extend(y_q_slice_prog.reshape(-1).tolist())
            #indexes_list_prog.extend(index_prog.reshape(-1).tolist())
            

            index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
            index_prog = index_prog.int()
            y_q_slice_prog = y_slice_prog - mu_prog 
            y_q_slice_prog = y_q_slice_prog*block_mask
            y_q_string  = self.gaussian_conditional_prog.compress(y_q_slice_prog, index_prog)
            y_hat_slice_prog = self.gaussian_conditional_prog.decompress(y_q_string, index_prog)
            y_hat_slice_prog = y_hat_slice_prog + mu_prog
            progressive_strings.append(y_q_string)
            

            y_hat_slice = self.merge(y_hat_slice_main,y_hat_slice_prog)

            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        #encoder_prog.encode_with_indexes(symbols_list_prog, indexes_list_prog, cdf_prog, cdf_lengths_prog, offsets_prog)

        #y_string_prog = encoder_prog.flush()
        #y_strings_prog.append(y_string_prog)

        #if q == 0:
        #    return {"strings": [y_strings, z_strings], "shape": [z.size()[-2:]]}
        #return {"strings": [y_strings, z_strings, z_string_prog,progressive_strings],  #preogressive_strings
        #        "shape": [z.size()[-2:],z_prog.size()[-2:]],          
        #        }   
        return {"strings": [y_strings, z_strings, z_string_prog, progressive_strings],  #preogressive_strings
                "shape": [z.size()[-2:],z_prog.size()[-2:]],          
                }   

    def decompress(self, strings, shape, quality):

        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        #y_string_prog = strings[-1][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_string)


        #cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        #cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        #offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        #decoder_prog = RansDecoder()
        #decoder_prog.set_stream(y_string_prog)

        z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])#strings[-1]  #self.entropy_bottleneck.decompress(strings[-1],shape[-1])
            
        latent_scales_prog = self.h_scale_prog(z_hat_prog)
        latent_means_prog = self.h_mean_prog(z_hat_prog)

        progressive_strings = strings[-1]

        mask = self.extract_mask(scale = latent_scales_prog, pr =q)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)

        y_hat_slices = []
        y_hat_progressive = []

        for slice_index in range(self.num_slices):

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

            #y_hat_slices.append(y_hat_slice)              
            block_mask = mask_slices[slice_index]

            mean_support_prog = torch.cat([latent_means_prog] + support_slices, dim=1)
            scale_support_prog = torch.cat([latent_scales_prog] + support_slices, dim=1)
            mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                            scale_support_prog, 
                                                            slice_index,
                                                            y_shape,
                                                            prog = True)

            index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask)
            index_prog = index_prog.int()  

            #rv_prog = decoder_prog.decode_stream(index_prog.reshape(-1).tolist(),
            #                                    cdf_prog, 
            #                                    cdf_lengths_prog, 
            #                                    offsets_prog)
            #rv_prog = torch.Tensor(rv_prog).reshape(mu_prog.shape)
            #y_hat_slice_prog = self.gaussian_conditional.dequantize(rv_prog, mu_prog)

            pr_strings = progressive_strings[slice_index]
            rv_prog = self.gaussian_conditional_prog.decompress(pr_strings, index_prog, means= mu_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
            y_hat_slice_prog = rv_prog.reshape(mu_prog.shape).to(mu_prog.device)

            y_hat_slice = self.merge(y_hat_slice_main,y_hat_slice_prog)

            
            
            y_hat_progressive.append(y_hat_slice_prog)
            y_hat_slices.append(y_hat_slice)

        if self.joiner_policy != "concatenation":
            y_hat = torch.cat(y_hat_slices, dim=1)
        else: 
            y_hat = torch.cat(y_hat_slices + y_hat_progressive, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}            






    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))

        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_a_prog: ",sum(p.numel() for p in self.h_a_prog.parameters()))

        print(" h_means_s: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_s: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print(" h_means_s_prog: ",sum(p.numel() for p in self.h_mean_prog.parameters()))
        print(" h_scale_s_prog: ",sum(p.numel() for p in self.h_scale_prog.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
        print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))

        if self.mask_policy== "learnable-mask":
            print("mask conv",sum(p.numel() for p in self.mask_conv.parameters()))
        

        if self.joiner_policy == "conditional":
            print("joiner",sum(p.numel() for p in self.joiner.parameters()))


        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


