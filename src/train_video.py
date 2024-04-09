
import wandb
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.optimizers import net_aux_optimizer
from compress.utils.functions import  create_savepath

from compress.training.video.step import train_one_epoch, valid_epoch, test_epoch
from compress.training.video.video_loss import collect_likelihoods_list, RateDistortionLoss
from compress.datasets import Vimeo90kDataset
from compress.utils.parser import parse_args_video
from compress.models import video_models
import os

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])



def save_checkpoint(state, is_best, last_pth,very_best):


    if is_best:
        print("ohhuuuuuuuuuuuuuu veramente il best-------------Z ",very_best)
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def main(argv):
    args = parse_args_video(argv)


    wandb.init(config= args, project="ssf-video", entity="albipresta") 
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Warning, the order of the transform composition should be kept.
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(args.patch_size)])
    valid_transforms = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(args.patch_size)])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = Vimeo90kDataset(
        args.dataset,
        split="train",
        transform=train_transforms,
    )
    valid_dataset =Vimeo90kDataset(
        args.dataset,
        split="valid",
        transform=valid_transforms,
    )
    test_dataset = Vimeo90kDataset(
        args.dataset,
        split="test",
        transform=test_transforms,
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )


    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )





    if args.model == "ssf":
        net = video_models[args.model]()
        net = net.to(device)
    
    else:
        pass #todo 



    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, return_details=True)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    counter = 0
    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        counter = train_one_epoch(
            counter,
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )

        start_val = time.time()
        loss = valid_epoch(epoch, valid_dataloader, net, criterion)
        end_val = time.time()
        print("Runtime of validating on epoch:  ", epoch)
        sec_to_hours(end_val - start_val)
        lr_scheduler.step(loss)

        start_test = time.time()
        _ = test_epoch(epoch, test_dataloader, net, criterion)
        end_test = time.time()
        print("Runtime of testing on epoch:  ", epoch)
        sec_to_hours(end_test - start_test)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)




        name_folder = "v" + args.code + "_" + args.model
        cartella = os.path.join(args.save_path,name_folder)


        if not os.path.exists(cartella):
            os.makedirs(cartella)
            print(f"Cartella '{cartella}' creata con successo.")  #ddddfffffrirririririr
        else:
            print(f"La cartella '{cartella}' esiste già.")


        last_pth, very_best =  create_savepath(cartella)



        #if is_best is True or epoch%10==0 or epoch > 98: #args.save:
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer":aux_optimizer if aux_optimizer is not None else "None",
                    "args":args
      
                },
                is_best,
                last_pth,
                very_best
                )

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
    main(sys.argv[1:])


