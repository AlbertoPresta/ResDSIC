import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F 
import time
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
from compress.result_list import *
import seaborn as sns

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
# Model Variants
from compress.utils.parser import parse_args_eval
from compress.utils.functions import AverageMeter, read_image
from compressai.ops import compute_padding

def evaluate_memory(model, input,p,mask_pol,device = "cuda"):
    with torch.inference_mode():
        x = read_image(input).to(device)
        nome_immagine = input.split("/")[-1].split(".")[0]
        x = x.unsqueeze(0) 
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)
        data =  model.compress(x_padded, quality =p, mask_pol = mask_pol )
        with profile(activities=[ProfilerActivity.CUDA],profile_memory=True, record_shapes=True) as prof:
            out_dec = model.decompress(data["strings"], data["shape"], quality = p, mask_pol = mask_pol)


        time.sleep(3)
        f = open("memory.txt", "a")
        f.write(prof.key_averages().table())
        print("olaaaaa memory: ",prof.key_averages().table())
        f.close()
            


def evaluate_speed(model, dataloader, device):
    times = []

    with torch.inference_mode():
        for input in dataloader:
            input = input.to(device)
            start_time = time.time()
            
            out = model(input)

            times.append(time.time()-start_time)
        
    print(f'Average Inference time on {len(times)} inputs is: {sum(times)/len(times)} Seconds')

def evaluate_flops(model, input,device,p,mask_pol):
    x = read_image(input).to(device)

    x = x.unsqueeze(0) 
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 25555
    x_padded = F.pad(x, pad, mode="constant", value=0)
    data =  model.compress(x_padded, quality =p, mask_pol = mask_pol )
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True) as prof:
            
            
            out_dec = model.decompress(data["strings"], data["shape"], quality = p, mask_pol = mask_pol)

        fw_flops = sum([int(evt.flops) for evt in prof.events()]) 
        print("-----> FLOPS: ",fw_flops)

    return fw_flops/(10**6)

def main(argv):

    psnr_res = {}
    bpp_res = {}
    decoded_time = {}

    name_dict = {"res_m4_pret_005_05_memd_frozen":"m4_pret_frozen",
                 "res_m4_005_06_encdec_blocked_kd9":"m4_pret_frozen_kd",
                 "res_m4_005_05_encdec":"m4_memd_005",
                 "res_m4_0025_05_encdec":"m4_memd_0025",
                 "res_m4_0035_05_encdec_k9":"me_memd_kd",
                 "res_m2_md":"m2_md",
                 "me_s_m4__0.005_0.05_":"me_005"}  
    


    args = parse_args_eval(argv)
    print(args)
    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"
    total_path = args.path + args.checkpoint[0] + args.model
    print("Loading", total_path)
    checkpoint = torch.load(total_path, map_location=device)
    new_args = checkpoint["args"]

    if "multiple_encoder" not in new_args:
        new_args.multiple_encoder = False
    else: 
        print("l'args ha il multiple encoder!")



    if "multiple_hyperprior" not in new_args:
        new_args.multiple_hyperprior = False
    else: 
        print("l'args ha il multiple_hyperprior!")


    if "double_dim" not in new_args:
        new_args.double_dim = False
    else: 
        print("l'args ha il multiple_hyperprior!")


    lmbda_list = new_args.lmbda_list
    wandb.init( config= args, project="COMPLEXITY", entity="albipresta")  #dddd 

    if new_args.seed is not None:
        torch.manual_seed(new_args.seed)
        random.seed(new_args.seed)
    

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path


    net = get_model(new_args,device, lmbda_list)

    #net = WACNN()
    
    
    checkpoint_model = replace_keys(checkpoint["state_dict"],multiple_encoder=new_args.multiple_encoder)
    net.load_state_dict(checkpoint_model ,strict = True) 
    #net.load_state_dict(checkpoint,strict = True)
    net.update() 

    p = 5 
    mask_pol ="point-based-std"
    input = filelist[0]

    print("INPUT: ",input)


    evaluate_memory(net, input,p,mask_pol,device = "cuda")

    print("memory evaluated")
    flops = evaluate_flops(net, input,device,p,mask_pol)



if __name__ == "__main__":  
    main(sys.argv[1:])

