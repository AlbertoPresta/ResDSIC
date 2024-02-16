

import argparse
from compress.zoo import models

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--aux-learning-rate",default=1e-3,type=float,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument( "--batch_size", type=int, default=16, help="Batch size (default: %(default)s)")

    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default = "/scratch/ResDSIC/starting_models/indcomlrp/last.pth.tar",help="Path to a checkpoint") #"/scratch/universal-dic/weights/q1/model.pth"
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--continue_training", action="store_true", help="continue training of the checkpoint")

    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    
    parser.add_argument("-e","--epochs",default=120,type=int,help="Number of epochs (default: %(default)s)",)



    parser.add_argument("--freeze", action="store_true", help="Use cuda") #ssss
    parser.add_argument("--frozen_base", action="store_true", help="Use cuda")


    parser.add_argument("-ilh","--independent_latent_hyperprior", action="store_true", help="Use cuda")
    parser.add_argument("-ibh","--independent_blockwise_hyperprior", action="store_true", help="Use cuda")
    parser.add_argument("-ilrp","--independent_lrp", action="store_true", help="use common lrp for progressive")

    parser.add_argument("--joiner_policy", type=str, default="concatenation", help="Batch size (default: %(default)s)")
    
    parser.add_argument("--lrp_prog", action="store_true", help="use common lrp for progressive")
    parser.add_argument("--lambda_list", nargs='+', type=float, default = [0.0035,  0.065])
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)

    parser.add_argument("--M", type=int, default=320, help="Batch size (default: %(default)s)")
    parser.add_argument("--mask_policy", type=str, default="two-levels", help="Batch size (default: %(default)s)") #sssswssss
    parser.add_argument("-m","--model",default="independent",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--num_images", type=int, default=300000, help="Batch size (default: %(default)s)")
    parser.add_argument("--N", type=int, default=192, help="Batch size (default: %(default)s)")
    parser.add_argument("--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)")

    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)

    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="/scratch/ResDSIC/models/", help="Where to Save model")
    

    parser.add_argument("--tester", action="store_true", help="use common lrp for progressive")
    parser.add_argument("--test_batch_size",type=int,default=1,help="Test batch size (default: %(default)s)",)

    parser.add_argument( "--valid_batch_size", type=int, default=16, help="Test batch size (default: %(default)s)",)
    
    
    
    parser.add_argument("--wandb_name", type=str, default="ResDSIC-MultiLevel", help="Batch size (default: %(default)s)")



    
    


    
    
    

    args = parser.parse_args(argv) #ssss
    return args