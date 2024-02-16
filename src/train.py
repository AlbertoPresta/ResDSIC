# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import wandb
import random
import sys
from compress.utils.functions import  create_savepath, set_seed
from compress.utils.parser import parse_args
from compress.utils.plot import plot_rate_distorsion
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   compress.training.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.loss import ScalableRateDistortionLoss, RateDistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import  configure_model
import os
from collections import OrderedDict

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])






class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }




    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

import copy


def save_checkpoint(state, is_best, filename,filename_best,very_best):


    if is_best:
        print("+ veramente il best-------------Z ",very_best)
        torch.save(state, filename_best)
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, filename)






def main(argv):
    args = parse_args(argv)
    

    wandb_name = args.wandb_name 

    wandb.init(project = wandb_name, entity="albipresta")  #dddddd 
    print("residual arguments!!!!-----> ",args)

    #torch.autograd.set_detect_anomaly(True)

    set_seed(args.seed)
    #if args.seed is not None:
        #torch.manual_seed(args.seed)
        #random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(args.dataset, split="test", transform=train_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak", transform= test_transforms)

    filelist = test_dataset.image_path

    device = "cuda" #if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)



    net = configure_model(args)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)



    last_epoch = 0
    if args.checkpoint != "none":  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = 0 # checkpoint["epoch"] + 1
        #net.update()

        #net.entropy_bottleneck._quantized_cdf = checkpoint["state_dict"]["entropy_bottleneck._offset"].to(device)
        #net.entropy_bottleneck_prog._quantized_cdf = checkpoint["state_dict"]["entropy_bottleneck_prog._offset"].to(device)

        
        del checkpoint["state_dict"]["entropy_bottleneck_prog._offset"]
        del checkpoint["state_dict"]["entropy_bottleneck_prog._cdf_length"]
        del checkpoint["state_dict"]["entropy_bottleneck_prog._quantized_cdf"]

        del checkpoint["state_dict"]["entropy_bottleneck._offset"]
        del checkpoint["state_dict"]["entropy_bottleneck._cdf_length"]
        del checkpoint["state_dict"]["entropy_bottleneck._quantized_cdf"]
        

        del checkpoint["state_dict"]["gaussian_conditional_prog._offset"]
        del checkpoint["state_dict"]["gaussian_conditional_prog._cdf_length"]
        del checkpoint["state_dict"]["gaussian_conditional_prog._quantized_cdf"]
        del checkpoint["state_dict"]["gaussian_conditional_prog.scale_table"]

        del checkpoint["state_dict"]["gaussian_conditional._offset"]
        del checkpoint["state_dict"]["gaussian_conditional._cdf_length"]
        del checkpoint["state_dict"]["gaussian_conditional._quantized_cdf"]
        del checkpoint["state_dict"]["gaussian_conditional.scale_table"]
        
        

        net.load_state_dict(checkpoint["state_dict"],strict = False)
        
        net.update()
        print("PROVA HO FATTO IL LOAD, SPERIAMO TUTTO BENE!!!!")


        
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)


    #if args.model != "conditional":
    #    criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list, frozen_base = args.frozen_base)
    #else:
    #criterion = RateDistortionLoss(lmbda=args.lmbda_list[-1])
    criterion = ScalableRateDistortionLoss(lmbda_list = args.lambda_list)

    if args.checkpoint != "none" and args.continue_training:    
        print("conitnuo il training!")
        optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])  #ddddd

    


    if args.frozen_base:
        print("entro su freezer!")
        net.freezer(total = False)






    best_loss = float("inf")
    counter = 0
    epoch_enc = 0
    #net.freezer()




    if args.tester: 
        #net.freezer(total = True)
        for p in net.parameters():
            p.requires_grad = False
        
        net.print_information()

        bpp_test, psnr_test = test_epoch(0, test_dataloader,criterion, net, pr_list = [0,1])
        print("test:  ",bpp_test,"   ",psnr_test)
        bpp, psnr = compress_with_ac(net,  filelist, device, epoch = 0, pr_list = [0,1],   writing = None)
        print("*********************************   OVER *********************************************************")
        print(bpp,"  ++++   ",psnr)
        return 0



    for epoch in range(last_epoch, args.epochs):
        print("******************************************************") #ffffff
        print(net.mask_policy)
        print("epoch: ",epoch)
        start = time.time()
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        num_tainable = net.print_information()
        #num_tainable = 1
        if num_tainable > 0:
            counter = train_one_epoch( net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, counter)
        



            
        print("finito il train della epoca")
        lista = [i  for i in range(len(net.lmbda_list))]
        loss = valid_epoch(epoch, valid_dataloader,criterion, net, pr_list = lista)
        print("finito il valid della epoca")

        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')

        bpp_test, psnr_test = test_epoch(epoch,
                                        test_dataloader,
                                        criterion,
                                        net,
                                        pr_list = [0.0,0.2,0.4,0.6,0.8,1.0], 
                                        mask_pol ="point-based-std" )
        print("finito il test della epoca")

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)


        
        if epoch%5==0 or is_best:

            net.update()
            if epoch_enc < -1:
                mask_pol = None 
                lista = [i  for i in range(len(net.lmbda_list))]
            else: 
                mask_pol = "point-based-std"
                lista = [0.0,0.2,0.4,0.6,0.8,1.0]

            bpp, psnr = compress_with_ac(net,  filelist, device, epoch = epoch_enc, pr_list = lista,   writing = None, mask_pol = mask_pol)
            print("total: ",bpp,"  ",psnr)
            psnr_res = {}
            bpp_res = {}

            bpp_res["our"] = bpp_test
            psnr_res["our"] = psnr_test

            psnr_res["base"] =   [29.20, 30.57,32.26,34.15,35.91,37.70]
            bpp_res["base"] =  [0.13,0.199,0.309,0.449,0.649,0.895]

            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc)
            epoch_enc += 1
        
        



        if args.checkpoint != "none":
            check = "pret"
        else:
            check = "zero"

        name_folder = check + "_" + "_multi_" + str(len(args.lambda_list)) + "_" + args.model + "_" + args.mask_policy + "_" + str(args.M) + \
                "_" + str(args.independent_latent_hyperprior)  + \
                "_" + str(args.independent_blockwise_hyperprior) + "_" + str(args.independent_lrp)
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.") 
        else:
            print(f"La cartella '{cartella}' esiste già.")



        filename, filename_best, very_best =  create_savepath(args, epoch, cartella)



        if is_best is True or epoch%10==0 or epoch > 98: #args.save:
            print("io qua devo entrare però!!!")

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer":aux_optimizer if aux_optimizer is not None else "None",
                    "args":args,
                    "epoch":epoch
                },
                is_best,
                filename,
                filename_best,
                very_best
                )

        print("log also the current leraning rate")

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr']
        #"train/beta": annealing_strategy_gaussian.bet
        }

        wandb.log(log_dict)



        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch #sss
     
    main(sys.argv[1:])
