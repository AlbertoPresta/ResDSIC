import torch 
import numpy as np 
from collections import OrderedDict


def initialize_model_from_pretrained( checkpoint,multiple_hyperprior, 
                                     checkpoint_enh = None):

    sotto_ordered_dict = OrderedDict()

    for c in list(checkpoint.keys()):
        if "g_s" in c: 

            nuova_stringa = "g_s.0." + c[4:]
            sotto_ordered_dict[nuova_stringa] = checkpoint[c]

        elif "g_a" in c: 
            nuova_stringa = "g_a.0." + c[4:]
            sotto_ordered_dict[nuova_stringa] = checkpoint[c]
        elif "cc_" in c or "lrp_" in c or "gaussian_conditional" or "entropy_bottleneck" in c: 
            sotto_ordered_dict[c] = checkpoint[c]
        else:
            continue


    for c in list(sotto_ordered_dict.keys()):
        if "h_scale_s" in c or "h_a" in c  or "h_mean_s" in c:
            sotto_ordered_dict.pop(c)
    if multiple_hyperprior:
        for c in list(checkpoint.keys()):
            if "h_mean_s" in c:
                nuova_stringa = "h_mean_s.0." + c[9:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]     
            elif "h_scale_s" in c:
                nuova_stringa = "h_scale_s.0." + c[10:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]   


    for c in list(sotto_ordered_dict.keys()):
        if "h_a" in c:
            sotto_ordered_dict.pop(c)


    if checkpoint_enh is not None:
        print("prendo anche il secondo modello enhanced")
        for c in list(checkpoint_enh.keys()):
            if "g_s" in c: 
                nuova_stringa = "g_s.1." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint_enh[c]

            else:
                continue




    return sotto_ordered_dict
    


