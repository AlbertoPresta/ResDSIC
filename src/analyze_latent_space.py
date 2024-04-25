import wandb
import random
import sys
import matplotlib.pyplot as plt
from compress.utils.parser import parse_args_eval
import torch
from   compress.training.image.step import  compress_with_ac
from torchvision import transforms
from compress.datasets.utils import  TestKodakDataset
from compress.models import get_model
from compress.utils.plot import plot_rate_distorsion, plot_decoded_time
import numpy as np
import os
from compress.result_list import *
import torch.nn.functional as F 
from compressai.ops import compute_padding, ste_round
from compress.utils.functions import AverageMeter, read_image
from compress.models import ChannelProgresssiveWACNN, WACNN, initialize_model_from_pretrained
import seaborn as sns
from torchvision.utils import save_image
palette = sns.color_palette("tab10")


torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

from collections import OrderedDict


def replace_keys(checkpoint, multiple_encoder):
    # Creiamo un nuovo OrderedDict con le chiavi modificate all'interno di un ciclo for
    nuovo_ordered_dict = OrderedDict()
    for chiave, valore in checkpoint.items():
        if multiple_encoder:
            if "g_a_enh." in chiave: 
                
                nuova_chiave = chiave.replace("g_a_enh.", "g_a.1.")
                nuovo_ordered_dict[nuova_chiave] = valore
            elif "g_a." in chiave and "g_a.0.1.beta" not in list(checkpoint.keys()): 
                nuova_chiave = chiave.replace("g_a.", "g_a.0.")
                nuovo_ordered_dict[nuova_chiave] = valore  
            else: 
                nuovo_ordered_dict[chiave] = valore   
        else:
            nuovo_ordered_dict[chiave] = valore  
    return nuovo_ordered_dict   




def analyze_latent_space(net,  filelist, device, quality ,mask_pol, nome_cartella):

    x = read_image(filelist).to(device)

    x = x.unsqueeze(0) 
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)

    
    if net.multiple_encoder is False:
        y = net.g_a(x)
        y_base = y 
        y_enh = y
    else:
        y_base = net.g_a[0](x)
        y_enh = net.g_a[1](x)
        y = torch.cat([y_base,y_enh],dim = 1).to(x.device) #dddd
    y_shape = y.shape[2:]
    latent_means, latent_scales, z_likelihoods = net.compute_hyperprior(y, quality)

    y_slices = y.chunk(net.num_slices, 1) # total amount of slicesy,

    y_hat_slices = []
    y_likelihood = []


    for slice_index in range(net.num_slice_cumulative_list[0]):
        y_slice = y_slices[slice_index]
        idx = slice_index%net.num_slice_cumulative_list[0]
        indice = min(net.max_support_slices,idx)
        support_slices = (y_hat_slices if net.max_support_slices < 0 else y_hat_slices[:indice]) 
            
        mean_support = torch.cat([latent_means[:,:net.division_dimension[0]]] + support_slices, dim=1)
        scale_support = torch.cat([latent_scales[:,:net.division_dimension[0]]] + support_slices, dim=1) 

            
        mu = net.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]  
        scale = net.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        _, y_slice_likelihood = net.gaussian_conditional(y_slice, scale, mu, training = False)
        y_hat_slice = ste_round(y_slice - mu) + mu

        # METTERE QUA
        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(slice_index),"like_base")
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(y_slice_likelihood,os.path.join(cartella_blocco,"ch_" + str(slice_index) + ".png")) 


        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(slice_index),"std_base")
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(scale,os.path.join(cartella_blocco,"ch_" + str(slice_index) + ".png")) 

        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(slice_index),"mu_base")
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(mu,os.path.join(cartella_blocco,"ch_" + str(slice_index) + ".png")) 


        lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
        lrp = net.lrp_transforms[idx](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice += lrp  

        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"base")
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(y_hat_slice,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png"))              
    if quality == 0: #and  slice_index == self.num_slice_cumulative_list[0] - 
        return 0
            

    y_hat_slices_quality = []

    y_likelihood_quality = []
    y_likelihood_quality = y_likelihood + []



    
    for slice_index in range(net.ns0,net.ns1):

        y_slice = y_slices[slice_index]
        current_index = slice_index%net.ns0


        if net.delta_encode:
            y_slice = y_slice - y_slices[current_index] 

            
        support_slices = net.determine_support(y_hat_slices,
                                                    current_index,
                                                    y_hat_slices_quality) 
            
        mean_support = torch.cat([latent_means[:,net.division_dimension[0]:]] + support_slices, dim=1)
        scale_support = torch.cat([latent_scales[:,net.division_dimension[0]:]] + support_slices, dim=1)  

        mu = net.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]  

        scale = net.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]



        sc_base = torch.cat([scale,y_hat_slices[current_index]],dim = 1) #fff
        block_mask = net.masking(scale,
                                      scale_base = sc_base ,
                                      slice_index = current_index, 
                                      pr = quality,
                                        mask_pol = mask_pol) #scale, slice_index = 0,  pr = 0, mask_pol = None
        block_mask = net.masking.apply_noise(block_mask, False)


        y_slice_m = y_slice  - mu
        y_slice_m = y_slice_m*block_mask

        _, y_slice_likelihood = net.gaussian_conditional(y_slice_m, scale*block_mask, training = False)
        y_hat_slice = ste_round(y_slice - mu)*block_mask + mu




        y_likelihood_quality.append(y_slice_likelihood)




        # METTERE QUA
        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"like_" + str(quality))
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(y_slice_likelihood,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png")) 


        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"std_"+ str(quality))
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(scale,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png")) 

        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"mu_"+ str(quality))
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(mu,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png")) 

        if net.residual_before_lrp:
            y_hat_slice = net.merge(y_hat_slice,y_hat_slices[current_index],current_index)


        lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
        lrp = net.lrp_transforms_prog[current_index](lrp_support)
        lrp = 0.5 * torch.tanh(lrp)
        y_hat_slice += lrp   

        # METTERE QUA
        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"enh_"+str(quality))
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(y_hat_slice,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png")) 

        # faccio il merge qua!!!!!
        if net.residual_before_lrp is False:
            y_hat_slice = net.merge(y_hat_slice,y_hat_slices[current_index],current_index)   #ddd

        y_hat_slices_quality.append(y_hat_slice)    

        # METTERE QUA
        cartella_blocco = os.path.join(nome_cartella,"blocks","block_" + str(current_index),"sum_"+str(quality))
        os.makedirs(cartella_blocco) if  os.path.exists(cartella_blocco) else print("ciao")
        save_image(y_hat_slice,os.path.join(cartella_blocco,"ch_" + str(current_index) + ".png")) 
    return 0


def main(argv):
    args = parse_args_eval(argv)


    if args.cluster == "hssh":

        kodak_path = "/scratch/dataset/kodak"
        save_path = "/scratch/ResDSIC/models/"

    elif args.cluster == "nautilus":
        kodak_path = "/data/kodak"
        save_path = "/data/latents"
    elif args.cluster == "ucsd":
        kodak_path = "/data/alberto/kodak"
        save_path = "/data/alberto/latents"
        path = "/data/alberto/resDSIC/"        
    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"
    total_path = path + args.checkpoint[0] + args.model
    print("Loading", total_path)
    checkpoint = torch.load(total_path, map_location=device)
    new_args = checkpoint["args"]

    if "multiple_encoder" not in new_args:
        new_args.multiple_encoder = False
    if "multiple_hyperprior" not in new_args:
        new_args.multiple_hyperprior = False
    if "double_dim" not in new_args:
        new_args.double_dim = False



    lmbda_list = new_args.lmbda_list
    #wandb.init( config= args, project="EVAL", entity="albipresta")  #dddd 

    if new_args.seed is not None:
        torch.manual_seed(new_args.seed)
        random.seed(new_args.seed)
    

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )
    test_dataset = TestKodakDataset(data_dir=kodak_path, transform= test_transforms)

    filelist = test_dataset.image_path




    net = get_model(new_args,device, lmbda_list)

    #net = WACNN()
    
    
    checkpoint_model = replace_keys(checkpoint["state_dict"],multiple_encoder=new_args.multiple_encoder)
    net.load_state_dict(checkpoint_model ,strict = True) 
    #net.load_state_dict(checkpoint,strict = True)
    net.update() 

    net.print_information()
    mask_pol = "point-based-std"
    qualities = [0,1,10]

    
    for i,kod in enumerate(filelist):
        nome_cartella = os.path.join(save_path,kod)
        os.makedirs(nome_cartella) if not nome_cartella else print("cartella già esistente")
        os.makedirs(os.path.join(nome_cartella,"rec")) if not os.path.join(nome_cartella,"rec")\
                                                         else print("cartella già esistente")

        os.makedirs(os.path.join(nome_cartella,"blocks")) if not os.path.join(nome_cartella,"rec")\
                                                         else print("cartella già esistente")
        for q in qualities:
           c = analyze_latent_space(net, kod, device, q, mask_pol, nome_cartella)




if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    main(sys.argv[1:])
