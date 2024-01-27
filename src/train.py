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
from compress.utils.functions import  create_savepath
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
from compress.zoo import models
import os

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

    print("dio porco: ",len(aux_parameters))
    print("zio porco: ",aux_parameters)



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




def save_checkpoint(state, is_best, filename,filename_best,very_best):


    if is_best:
        torch.save(state, filename_best)
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, filename)



def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

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

    if args.lmbda_list == []:
        lmbda_list = None
    else:
        lmbda_list = args.lmbda_list



    
    net = models[args.model]( N = args.N,
                             M = args.M,
                            scalable_levels = args.scalable_levels, 
                              mask_policy = args.mask_policy,
                              lmbda_list = lmbda_list,
                              lrp_prog = args.lrp_prog,
                              independent_lrp = args.ind_lrp
                             )
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)



    last_epoch = 0
    if args.checkpoint != "none":  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = 0 # checkpoint["epoch"] + 1
        net.update()
        net.load_state_dict(checkpoint)
        
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)


    if args.checkpoint != "none" and args.continue_training:    
        print("conitnuo il training!")
        optimizer.load_state_dict(checkpoint["optimizer"])
        #aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    


    if args.freeze:
        print("entro su freezer!")
        net.freezer()

    best_loss = float("inf")
    counter = 0
    epoch_enc = 0

    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        num_tainable = net.print_information()
        if num_tainable > 0:
            counter = train_one_epoch( net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, counter)
            
        print("finito il train della epoca")
        loss = valid_epoch(epoch, valid_dataloader,criterion, net, pr_list = net.lmbda_list)
        print("finito il valid della epoca")

        lr_scheduler.step(loss)
        print(f'Current patience level: {lr_scheduler.patience - lr_scheduler.num_bad_epochs}')

        _ = test_epoch(epoch, test_dataloader,criterion, net, pr_list = net.lmbda_list)
        print("finito il test della epoca")

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if epoch%5==0 or is_best:
            net.update()

            bpp, psnr = compress_with_ac(net,  filelist, device, epoch = epoch_enc, pr_list = net.lmbda_list,   writing = None)
            psnr_res = {}
            bpp_res = {}

            bpp_res["our"] = bpp
            psnr_res["our"] = psnr

            psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
            bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]

            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc)
            epoch_enc += 1



        if args.checkpoint != "none":
            check = "pret"
        else:
            check = "zero"
        # creating savepath
        name_folder = check + "_" + "_" + str(args.scalable_levels) + "_" + args.model + "_" + args.mask_policy + "_" + str(args.M) + "_" + str(args.N)  + "_" + str(args.lmbda_list[0]) + "_" + str(args.lmbda_list[-1]) +"_" + str(args.freeze) 
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.") 
        else:
            print(f"La cartella '{cartella}' esiste già.")



        filename, filename_best, very_best =  create_savepath(args, epoch, cartella)



        if (is_best is True and epoch > 10) or epoch%10==0: #args.save:
            print("io qua devo entrare però!!!")
            net.update()
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



        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch
    wandb.init(project="ResDSIC-zero-one", entity="albipresta")   
    main(sys.argv[1:])
