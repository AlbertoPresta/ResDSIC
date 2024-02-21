import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional #fff
from compress.layers import GDN
from ..utils import conv
from compress.ops import ste_round
from compress.layers import  Win_noShift_Attention
from ..cnn import WACNN
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



class ResWACNNSharedEntropy(WACNN):
    """CNN based model"""

    def __init__(self, N=192,
                M=320,
                mask_policy = "learnable-mask",
                lambda_list = [0.0035,0.065],
                
                **kwargs):
        super().__init__(N = N, M = M, **kwargs)


        assert lambda_list is not None 
        self.N = N 
        self.M = M 
        self.halve = 8
        self.level = 5 if self.halve == 8 else -1 
        self.factor = self.halve**2
        assert self.N%self.factor == 0 
        self.T = int(self.N//self.factor) + 3
        self.mask_policy = mask_policy

        self.scalable_levels = len(lambda_list)
        self.lmbda_list = lambda_list
        self.lmbda_index_list = dict(zip(self.lmbda_list, [i  for i in range(len(self.lmbda_list))] ))


        
        print(self.lmbda_index_list)
        print("questa è la lista finale",self.lmbda_index_list)
        
        self.entropy_bottleneck = EntropyBottleneck(self.N)
        self.entropy_bottleneck_prog = EntropyBottleneck(self.N) #utilizzo lo stesso modello, ma non lo stesso entropy bottleneck
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_prog = GaussianConditional(None)

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



    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """

        aux_loss1 = self.entropy_bottleneck_prog.loss()
        aux_loss2 = self.entropy_bottleneck.loss()
        aux_loss = aux_loss1 + aux_loss2 

        #aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss


    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))
        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        #print(" h_a_orog: ",sum(p.numel() for p in self.h_a_prog.parameters()))

        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))

        if self.mask_policy== "learnable-mask":
            print("mask conv",sum(p.numel() for p in self.mask_conv.parameters()))
           

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def split_ga(self, x, begin = True):
        if begin:
            layers_intermedi = list(self.g_a.children())[:self.level + 1]
        else:
            layers_intermedi = list(self.g_a.children())[self.level + 1:]
        modello_intermedio = nn.Sequential(*layers_intermedi)
        return modello_intermedio(x)

    def freezer(self):

        for p in self.parameters():
            p.requires_grad = False 
        for _,p in self.named_parameters():
            p.requires_grad = False
        
        if self.mask_policy in  ("all-one","two-levels","learnable-mask"):

            for p in self.g_a_progressive.parameters():
                p.requires_grad = True
            
            for p in self.entropy_bottleneck_prog.parameters():
                p.requires_grad = True 
            for _,p in self.entropy_bottleneck_prog.named_parameters():
                p.requires_grad = True


            for p in self.entropy_bottleneck.parameters():
                p.requires_grad = True 
            for _,p in self.entropy_bottleneck.named_parameters():
                p.requires_grad = True

            for p in self.h_a.parameters():
                p.requires_grad = True 

            for p in self.h_mean_s.parameters():
                p.requires_grad = True 

            for p in self.h_scale_s.parameters():
                p.requires_grad = True 

            for module in self.cc_mean_transforms:
                for p in module.parameters():
                    p.requires_grad = True 

            for module in self.cc_scale_transforms:
                for p in module.parameters():
                    p.requires_grad = True 

            if self.mask_policy != "all-one":
                for p in self.g_s.parameters():
                    p.requires_grad = True 


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = self.gaussian_conditional_prog.update_scale_table(scale_table, force=force)
        #updated |= super().update(force=force)

        print("UNO")
        self.entropy_bottleneck_prog.update()
        print("DUE")
        self.entropy_bottleneck.update()
        return updated


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
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            if pr == len(self.lmbda_list) -1:
                return torch.ones_like(scale).to(scale.device)
  
            importance_map =  self.mask_conv(scale) 
            importance_map = torch.clip(importance_map + 0.5, 0, 1)
            gamma = torch.sum(torch.stack([self.gamma[j] for j in range(self.lmbda_index_list[pr])]),dim = 0) # più uno l'hom esso in lmbda_index
            gamma = gamma[None, :, None, None]
            gamma = torch.relu(gamma)

            
            adjusted_importance_map = torch.pow(importance_map, gamma)
            return adjusted_importance_map          
        elif self.mask_policy == "all-one":
            return torch.ones_like(scale).to(scale.device)

        elif self.mask_policy == "all-zero":
            return torch.zeros_like(scale).to(scale.device)
        elif self.mask_policy == "two-levels":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            else:
                return torch.ones_like(scale).to(scale.device)
        else:
            raise NotImplementedError()



    def define_quality(self,quality):
        if quality is None:
            list_quality = self.lmbda_list 
        elif isinstance(quality,list):
            list_quality = quality 
        else:
            list_quality = [quality] 
        return list_quality
    
    def extract_mu_and_scale(self,mean_support, scale_support,slice_index,y_shape):

            if True: #prog is False:
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale = self.cc_scale_transforms[slice_index](scale_support)
                scale = scale[:, :, :y_shape[0], :y_shape[1]] 

                return mu, scale 
             
        
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
        
        z_prog = self.h_a(y_progressive) 
        _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog) # this is different (could have different dix)

        z_offset_prog = self.entropy_bottleneck_prog._get_medians()
        z_tmp_prog = z_prog - z_offset_prog
        z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

        scales_prog = self.h_scale_s(z_hat_prog)
        means_prog = self.h_mean_s(z_hat_prog)

        list_quality = self.define_quality(quality)        


        y_slices = y.chunk(self.num_slices, 1)

        y_likelihoods_progressive = []
        y_likelihood_main = []

        x_hat_progressive = []

        y_hats = []
        
        for j,p in enumerate(list_quality): 
            mask = self.extract_mask(latent_scales,pr = p)
            if self.mask_policy == "learnable-mask": # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
                if self.training:
                    samples = mask + (torch.rand_like(mask) - 0.5)
                    mask = samples + samples.round().detach() - samples.detach()  # Differentiable torch.round()   
                else:
                    mask = torch.round(mask)
                        
        


            y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)
            #mask_slices = mask.chunk(self.num_slices,dim = 1)


            y_hat_slices = []
            y_hat_complete = []
            y_hat_prog = []
            y_likelihood_prog = []

            

            for slice_index, y_slice in enumerate(y_slices):
                ####################################################################################################
                ##############################    PARTO DAL MAIN LATENT REPRESENTATION ##########################
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                
                # encode the main latent representation 
                mean_support = torch.cat([latent_means] + support_slices, dim=1)
                scale_support = torch.cat([latent_scales] + support_slices, dim=1)
                mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

                _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                
                
                if j == 0:
                    y_likelihood_main.append(y_slice_likelihood)
                y_hat_slice = ste_round(y_slice - mu) + mu 
                

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp

                y_hat_slices.append(y_hat_slice)


                #############################################################
                #######################   TOCCA ALLA PARTE PROGRESSIVE ######
                ############################################################
                if  self.lmbda_index_list[p] != 0:
                    y_prog_slice = y_progressive_slices[slice_index]
                    #block_mask = mask_slices[slice_index]
                    support_prog_slices = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])


                    #[latent_means] + support_slices
                    mean_support_prog = torch.cat([means_prog] + support_prog_slices, dim=1)
                    scale_support_prog = torch.cat([scales_prog] + support_prog_slices, dim=1)
                    mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                    scale_support_prog,
                                                                    slice_index,
                                                                    y_shape
                                                                    )

                    _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog,mu_prog)

                    y_likelihood_prog.append(y_slice_likelihood_prog)

                    y_hat_prog_slice = ste_round(y_prog_slice - mu_prog) + mu_prog

                    lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                    lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_prog_slice += lrp

                    y_hat_prog.append(y_hat_prog_slice)

                    y_hat_complete_slice = y_hat_slice + y_hat_prog_slice
                    y_hat_complete.append(y_hat_complete_slice)
                else:
                    y_hat_complete.append(y_hat_slice)




            y_hat_q = torch.cat(y_hat_complete,dim = 1) #questo va preso 
            x_hat_q = self.g_s(y_hat_q) 



            y_hats.append(y_hat_q.unsqueeze(0))
            x_hat_progressive.append(x_hat_q.unsqueeze(0)) 

            if  self.lmbda_index_list[p] != 0:
                y_likelihood_progressive = torch.cat(y_likelihood_prog,dim = 1)
                y_likelihoods_progressive.append(y_likelihood_progressive.unsqueeze(0))


        x_hat_progressive = torch.cat(x_hat_progressive,dim = 0) #num_scalable-1, BS,3,W,H 
        y_likelihoods = torch.cat(y_likelihood_main, dim = 0).unsqueeze(0) # 1,BS,3,W,H solo per base 

        
        if len(y_likelihoods_progressive) == 0:
            y_likelihoods_prog = torch.ones_like(y_likelihoods).to(y_likelihoods.device)
        else:
            y_likelihoods_prog = torch.cat(y_likelihoods_progressive,dim = 0)

        y_hat = torch.cat(y_hats,dim = 0)

        return {
            "x_hat": x_hat_progressive,

            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods,"z_prog":z_likelihoods_prog,"y_prog":y_likelihoods_prog},
            "y": y_hat, "z_hat_prog":z_hat_prog ,"z_hat":z_hat
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


        
        z_prog = self.h_a(y_progressive)

        z_string_prog = self.entropy_bottleneck_prog.compress(z_prog)
        z_hat_prog = self.entropy_bottleneck_prog.decompress(z_string_prog,z_prog.size()[-2:])


        latent_scales_prog = self.h_scale_s(z_hat_prog)
        latent_means_prog = self.h_mean_s(z_hat_prog)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        #cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        #cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        #offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        #encoder_prog = BufferedRansEncoder()

        symbols_list = []
        symbols_list_prog = []

        indexes_list = []
        indexes_list_prog = []

        
        y_strings = []
        y_strings_prog = []

        mask = self.extract_mask(latent_scales,pr = quality)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)
        y_hat_prog = []

        progressive_strings = []
        for slice_index, y_slice in enumerate(y_slices):
            #part con la parte main 
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale = self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)

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

            # now progressive
            if self.lmbda_index_list[quality] != 0:

                block_mask = mask_slices[slice_index]
                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                y_prog_slice = y_progressive_slices[slice_index]

                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                scale_support_prog,
                                                                slice_index,
                                                                y_shape
                                                                )





                index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog)
                #index_prog = index_prog.int()

                #y_q_prog_slice = y_prog_slice - mu_prog 
                #y_q_prog_slice = y_q_prog_slice#*block_mask

                

                y_q_string  = self.gaussian_conditional_prog.compress( y_prog_slice, index_prog, means=mu_prog)
                
                y_hat_prog_slice = self.gaussian_conditional_prog.decompress(y_q_string, index_prog, means= mu_prog)

                    

                #y_q_prog_slice = self.gaussian_conditional_prog.quantize( y_prog_slice, "symbols", means=mu_prog)
                #y_q_prog_slice = y_q_prog_slice.int()

                progressive_strings.append(y_q_string)

                lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_prog_slice += lrp

                y_hat_prog.append(y_hat_prog_slice)


        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        if self.lmbda_index_list[quality] == 0:
            return {"strings": [y_strings, z_strings], "shape": [z.size()[-2:]]}
    
        return {"strings": [y_strings, z_strings, z_string_prog, progressive_strings], 
                "shape": [z.size()[-2:],z_prog.size()[-2:]],          
                }
    

    def decompress(self, strings, shape, quality):

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[0])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        # extract the mask!
        mask = self.extract_mask(latent_scales,pr = quality)
        mask = torch.round(mask)
        mask_slices = mask.chunk(self.num_slices,dim = 1)

        y_string = strings[0][0]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)


        if self.lmbda_index_list[quality] != 0:
            z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])#strings[-1]  #self.entropy_bottleneck.decompress(strings[-1],shape[-1])
            
            latent_scales_prog = self.h_scale_s(z_hat_prog)
            latent_means_prog = self.h_mean_s(z_hat_prog)

            progressive_strings = strings[-1]

        y_hat_slices = []
        y_hat_prog = []
        y_hat_complete = []
        
        
        for slice_index in range(self.num_slices):
        
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale =  self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)
            
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets) #ddd
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            if self.lmbda_index_list[quality] != 0:
       
                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                
                
                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog,
                                                                 scale_support_prog,
                                                                 slice_index,
                                                                 y_shape
                                                                 )

                index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog)
                #index_prog = index_prog.int()

                pr_strings = progressive_strings[slice_index]
                rv = self.gaussian_conditional_prog.decompress(pr_strings, index_prog, means= mu_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                
                y_hat_prog_slice = rv.reshape(mu_prog.shape).to(mu_prog.device)
            
                lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_prog_slice += lrp

                y_hat_prog.append(y_hat_prog_slice)

                y_hat_complete_slice = y_hat_prog_slice + y_hat_slice
                y_hat_complete.append(y_hat_complete_slice)
            else:
                y_hat_complete.append(y_hat_slice)


        y_hat = torch.cat(y_hat_complete, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}