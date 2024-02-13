import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder

from ..utils import conv
from compress.ops import ste_round
from compress.layers import conv3x3, subpel_conv3x3
from .shared_entropy import ResWACNNSharedEntropy 
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ResWACNNIndependentEntropy(ResWACNNSharedEntropy):
    def __init__(self, 
                 N=192,
                M=320,
                mask_policy = "learnable-mask",
                lmbda_list = [0.05],
                lrp_prog = True,
                independent_lrp = False,
                **kwargs):
        super().__init__(N = N, 
                         M = M,
                         mask_policy=mask_policy,
                         lmbda_list=lmbda_list,
                         lrp_prog=lrp_prog,
                         independent_lrp=independent_lrp,
                          **kwargs)


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






        self.h_a_prog = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2), #dddd
        )

        self.h_mean_s_prog = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s_prog = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        
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
        


    def relu_clip(self,x, min = 0, max = 1):
        return torch.relu(x - min) - torch.relu(x - max)


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
            #importance_map = torch.clip(importance_map + 0.5, 0, 1)
            importance_map = torch.relu_clip(importance_map) 

            #importance_map = torch.sigmoid(importance_map)

            index_pr = self.scalable_levels -  pr
            index_pr = int(index_pr)
            #index_pr = pr - 1
            gamma = torch.sum(torch.stack([self.gamma[j] for j in range(index_pr)]),dim = 0) # più uno l'hom esso in lmbda_index
            gamma = gamma[None, :, None, None]
            gamma = torch.relu(gamma) 


            adjusted_importance_map = torch.pow(importance_map, gamma)
            #adjusted_importance_map.register_hook(lambda grad: print('adjusted_importance_map back', (grad != grad).any().item()))
            return adjusted_importance_map          

        elif self.mask_policy == "learnable-mask-nested":

            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            
            assert scale_prog is not None 
            scale_input = torch.cat([scale,scale_prog],dim = 1)

            importance_map = torch.sum(torch.stack([self.relu_clip(self.mask_conv[i](scale_input)) for i in range(pr)],dim = 0),dim = 0) 
            importance_map = self.relu_clip(importance_map)    

            return importance_map       



        elif self.mask_policy == "two-levels":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            else:
                return torch.ones_like(scale).to(scale.device)
        else:
            raise NotImplementedError()
        

    def print_information(self):
        print(" g_a: ",sum(p.numel() for p in self.g_a.parameters()))
        print(" g_a_progressive: ",sum(p.numel() for p in self.g_a_progressive.parameters()))

        print(" h_a: ",sum(p.numel() for p in self.h_a.parameters()))
        print(" h_a_prog: ",sum(p.numel() for p in self.h_a_prog.parameters()))

        print(" h_means_s: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_s: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print(" h_means_s_prog: ",sum(p.numel() for p in self.h_mean_s_prog.parameters()))
        print(" h_scale_s_prog: ",sum(p.numel() for p in self.h_scale_s_prog.parameters()))

        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        print("cc_mean_transforms_prog",sum(p.numel() for p in self.cc_mean_transforms_prog.parameters()))
        print("cc_scale_transforms_prog",sum(p.numel() for p in self.cc_scale_transforms_prog.parameters()))

        print("lrp_transform",sum(p.numel() for p in self.lrp_transforms.parameters()))

        print("entropy_bottleneck",sum(p.numel() for p in self.entropy_bottleneck.parameters() if p.requires_grad == True))
        print("entropy_bottleneck PROG",sum(p.numel() for p in self.entropy_bottleneck_prog.parameters() if p.requires_grad == True))

        if "learnable-mask" in self.mask_policy:
            print("mask conv",sum(p.numel() for p in self.mask_conv.parameters()))
        
        if self.independent_lrp:
             print("lrp_transform_prog",sum(p.numel() for p in self.lrp_transforms_prog.parameters()))

           

        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

             

    def freezer(self, total = True ):


        if total:
            for p in self.parameters():
                p.requires_grad = False
        
        for p in self.g_a.parameters():
            p.requires_grad = False 
        for p in self.h_a.parameters():
            p.requires_grad = False 
        for p in self.h_mean_s.parameters():
            p.requires_grad = False 
        for p in self.h_scale_s.parameters():
            p.requires_grad = False 

            
        for m in self.cc_mean_transforms:
            for p in m.parameters():
                p.requires_grad = False 
            
        for m in self.cc_scale_transforms:
            for p in m.parameters():
                p.requires_grad = False

        for m in self.lrp_transforms:
            for p in m.parameters():
                p.requires_grad = False
            

           

    # provo a tenere tutti insieme! poi vediamo 
    def forward(self, x, quality =None, training = True):

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
        _, z_likelihoods_prog = self.entropy_bottleneck_prog(z_prog) # this is different (could have different dix)

        z_offset_prog = self.entropy_bottleneck_prog._get_medians()
        z_tmp_prog = z_prog - z_offset_prog
        z_hat_prog = ste_round(z_tmp_prog) + z_offset_prog

        scales_prog = self.h_scale_s_prog(z_hat_prog)
        means_prog = self.h_mean_s_prog(z_hat_prog)

        list_quality = self.define_quality(quality)        


        y_slices = y.chunk(self.num_slices, 1)

        y_likelihoods_progressive = []
        y_likelihood_main = []

        x_hat_progressive = []

        y_hats = []
        
        for j,p in enumerate(list_quality): 

            if p in list(self.lmbda_index_list.keys()):
                quality = self.lmbda_index_list[p]
            else: 
                quality = p

            
            mask = self.extract_mask(latent_scales,scale_prog = scales_prog, pr =quality)
            if self.mask_policy == "learnable-mask": # and self.lmbda_index_list[p]!=0 and self.lmbda_index_list[p]!=len(self.lmbda_list) -1:
                if training:
                    samples = mask + (torch.rand_like(mask) - 0.5)
                    mask = samples + samples.round().detach() - samples.detach()  # Differentiable torch.round()   
                else:
                    mask = torch.round(mask)
                        
            y_progressive_slices = y_progressive.chunk(self.num_slices,dim = 1)
            mask_slices = mask.chunk(self.num_slices,dim = 1)
            
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
                if  quality != 0:
                    y_prog_slice = y_progressive_slices[slice_index]
                    block_mask = mask_slices[slice_index]
                    support_prog_slices = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])


                    #[latent_means] + support_slices
                    mean_support_prog = torch.cat([means_prog] + support_prog_slices, dim=1)
                    scale_support_prog = torch.cat([scales_prog] + support_prog_slices, dim=1)
                    mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                    scale_support_prog,
                                                                    slice_index,
                                                                    y_shape,
                                                                   prog = True
                                                                    )

                    _, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog,mu_prog)
                    #_, y_slice_likelihood_prog = self.gaussian_conditional_prog(y_prog_slice, scale_prog,mu_prog)

                    y_likelihood_prog.append(y_slice_likelihood_prog)

                    y_hat_prog_slice = ste_round(y_prog_slice - mu_prog)*block_mask + mu_prog

                    if self.lrp_prog:
                        lrp_support = torch.cat([mean_support_prog, y_hat_prog_slice], dim=1)
                        if self.independent_lrp:
                            lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                        else:
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

            if  quality != 0:
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



    #/scratch/ResDSIC/models/zero__2_independent_two-levels_320_192_0.0035_0.065_False/0210__lambda__0.075__epoch__91_best_.pth.tar



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


        cdf_prog = self.gaussian_conditional_prog.quantized_cdf.tolist()
        cdf_lengths_prog = self.gaussian_conditional_prog.cdf_length.reshape(-1).int().tolist()
        offsets_prog = self.gaussian_conditional_prog.offset.reshape(-1).int().tolist()
        encoder_prog = BufferedRansEncoder()


        symbols_list = []
        indexes_list = []
        y_strings = []


        symbols_list_prog = []
        indexes_list_prog = []
        y_strings_prog = []


        if quality in list(self.lmbda_index_list.keys()):
            q = self.lmbda_index_list[quality] 
        else:
            q = quality

        if q != 0:


            z_prog = self.h_a_prog(y_progressive)

            z_string_prog = self.entropy_bottleneck_prog.compress(z_prog)
            z_hat_prog = self.entropy_bottleneck_prog.decompress(z_string_prog,z_prog.size()[-2:])

            latent_scales_prog = self.h_scale_s_prog(z_hat_prog)
            latent_means_prog = self.h_mean_s_prog(z_hat_prog)

            mask = self.extract_mask(latent_scales,scale_prog = latent_scales_prog, pr =q)
            mask = torch.round(mask)
            mask_slices = mask.chunk(self.num_slices,dim = 1)

            progressive_strings = []




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
            if q != 0:

                block_mask = mask_slices[slice_index]
                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                y_slice_prog = y_progressive_slices[slice_index]

                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog, 
                                                                scale_support_prog,
                                                                slice_index,
                                                                y_shape,
                                                                prog = True)


                index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask) #saràda aggiungere la maschera
                index_prog = index_prog.int()

                y_q_slice_prog = y_slice_prog - mu_prog 
                y_q_slice_prog = y_q_slice_prog*block_mask


                y_q_slice_prog = self.gaussian_conditional_prog.quantize(y_q_slice_prog, "symbols")
                y_hat_slice_prog = y_q_slice_prog + mu_prog

                symbols_list_prog.extend(y_q_slice_prog.reshape(-1).tolist())
                indexes_list_prog.extend(index_prog.reshape(-1).tolist())
        
                #y_q_string  = self.gaussian_conditional_prog.compress(y_q_slice_prog, index_prog)
                #y_hat_slice_prog = self.gaussian_conditional_prog.decompress(y_q_string, index_prog)
                #progressive_strings.append(y_q_string)
 

 

                

                if self.lrp_prog:
                    lrp_support = torch.cat([mean_support_prog,y_hat_slice_prog], dim=1)
                    if self.independent_lrp:
                        lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                    else:
                        lrp = self.lrp_transforms[slice_index](lrp_support) ##dddd
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)


        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        if q == 0:
            return {"strings": [y_strings, z_strings], "shape": [z.size()[-2:]]}
    
        
        encoder_prog.encode_with_indexes(symbols_list_prog, indexes_list_prog, cdf_prog, cdf_lengths_prog, offsets_prog)
        y_string_prog = encoder_prog.flush()
        y_strings_prog.append(y_string_prog)
        print("finito encoding")
        
        return {"strings": [y_strings, z_strings, z_string_prog,y_strings_prog],  #preogressive_strings
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


        if q != 0:

            z_hat_prog =  self.entropy_bottleneck_prog.decompress(strings[2],shape[-1])#strings[-1]  #self.entropy_bottleneck.decompress(strings[-1],shape[-1])
            
            latent_scales_prog = self.h_scale_s_prog(z_hat_prog)
            latent_means_prog = self.h_mean_s_prog(z_hat_prog)

            #progressive_strings = strings[-1]
            #y_string_prog = strings[2][0]


            mask = self.extract_mask(latent_scales,scale_prog = latent_scales_prog, pr =q)
            mask = torch.round(mask)
            mask_slices = mask.chunk(self.num_slices,dim = 1)

        y_hat_slices = []
        y_hat_prog = []
        y_hat_complete = []
               
        for slice_index in range(self.num_slices):

           
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            mu, scale =  self.extract_mu_and_scale(mean_support, scale_support,slice_index,y_shape)
            
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp


            y_hat_slices.append(y_hat_slice)


            if q != 0:

                support_slices_prog = (y_hat_prog if self.max_support_slices < 0 else y_hat_prog[:self.max_support_slices])
                
                block_mask = mask_slices[slice_index]
                #[latent_means] + support_slices
                mean_support_prog = torch.cat([latent_means_prog] + support_slices_prog, dim=1)
                scale_support_prog = torch.cat([latent_scales_prog] + support_slices_prog, dim=1)
                mu_prog, scale_prog = self.extract_mu_and_scale(mean_support_prog,
                                                                 scale_support_prog,
                                                                 slice_index,
                                                                 y_shape,
                                                                 prog = True
                                                                 )

                index_prog = self.gaussian_conditional_prog.build_indexes(scale_prog*block_mask)
                index_prog = index_prog.int()


                rv_prog = decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                rv_prog = torch.Tensor(rv_prog).reshape(mu_prog.shape)
                y_hat_slice_prog = self.gaussian_conditional_prog.dequantize(rv_prog, mu_prog)



                #pr_strings = progressive_strings[slice_index]
                #rv_prog = self.gaussian_conditional_prog.decompress(pr_strings, index_prog, means= mu_prog) # decoder_prog.decode_stream(index_prog.reshape(-1).tolist(), cdf_prog, cdf_lengths_prog, offsets_prog)
                
                #y_hat_slice_prog = rv_prog.reshape(mu_prog.shape).to(mu_prog.device)
            
                if self.lrp_prog:
                    lrp_support = torch.cat([mean_support_prog, y_hat_slice_prog], dim=1)
                    if self.independent_lrp:
                        lrp = self.lrp_transforms_prog[slice_index](lrp_support)
                    else:
                        lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_slice_prog += lrp

                y_hat_prog.append(y_hat_slice_prog)
                y_hat_complete_slice = y_hat_slice_prog + y_hat_slice
                y_hat_complete.append(y_hat_complete_slice)
            else:
                y_hat_complete.append(y_hat_slice)


        y_hat = torch.cat(y_hat_complete, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}