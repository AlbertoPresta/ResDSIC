

import argparse
from compress.models import models

def parse_args_eval(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddd
    parser.add_argument("--checkpoint", type=str, default = "/scratch/ResDSIC/models/res2/_very_best.pth.tar",help="Path to a checkpoint")#zero__multi__0.005_0.05_progressive_scalable_res_False_False#/scratch/ResDSIC/models/zero__multi__0.0035_0.05_cond_ind_two-levels_Trueconcatenation_False/_very_best.pth.tar
    args = parser.parse_args(argv)
    return args



def parse_args_mask(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddd
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=140,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument( "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)


    parser.add_argument("--list_quality", nargs='+', type=float, default = [0.5])
    parser.add_argument( "--batch_size", type=int, default=16, help="Batch size (default: %(default)s)")

    parser.add_argument("--num_images", type=int, default=300000, help="Batch size (default: %(default)s)")

    parser.add_argument("--patience", type=int, default=4, help="Batch size (default: %(default)s)")#ddddddd

    parser.add_argument("--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)")
    parser.add_argument("--mask_policy", type=str, default = "single-learnable-mask-quantile")
    parser.add_argument("--valid_batch_size",type=int,default=16,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size (default: %(default)s)", )
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--pretrained", action="store_true", help="Use cuda")
    parser.add_argument("--only_mask", action="store_true", help="Use cuda")
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)

    parser.add_argument("--lambda_list",dest="lmbda_list", nargs='+', type=float, default = [ 0.0025,  0.015, 0.050])


    parser.add_argument("--save_path", type=str, default="/scratch/ResDSIC/models/", help="Where to Save model")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--sampling_training", action="store_true", help="Save model to disk")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default = "/scratch/ResDSIC/models/res2/_very_best.pth.tar")#/scratch/ResDSIC/models/zero__multi__0.0035_0.05_cond_ind_two-levels_Trueconcatenation_False/_very_best.pth.tar

    args = parser.parse_args(argv)
    return args


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddd
    parser.add_argument("-m","--model",default="restcm",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=140,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument( "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    #dddd

    parser.add_argument("--lambda_list",dest="lmbda_list", nargs='+', type=float, default = [ 0.005, 0.050])
    parser.add_argument("--division_dimension", nargs='+', type=int, default = [320, 640])
    parser.add_argument("--inner_dimensions", nargs='+', type=int, default = [192, 320])
    parser.add_argument("--list_quality", nargs='+', type=int, default = [0])
    parser.add_argument( "--batch_size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument( "--dim_chunk", type=int, default=32, help="Batch size (default: %(default)s)")

    parser.add_argument( "--gamma", type=float, default=0.5, help="gamma")
    parser.add_argument("--kd_base", action="store_true", help="KD base")


    parser.add_argument("--num_images", type=int, default=300000, help="Batch size (default: %(default)s)")

    parser.add_argument("--N", type=int, default=128, help="Batch size (default: %(default)s)")#ddddd#ddd
    parser.add_argument("--M", type=int, default=640, help="Batch size (default: %(default)s)")
    parser.add_argument("--patience", type=int, default=4, help="Batch size (default: %(default)s)")#ddddddd

    parser.add_argument("--type_loss", type=int, default=0, help="v")#ddddddd

    parser.add_argument("--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)")
    parser.add_argument("--mask_policy", type=str, default = "two-levels", help="Batch size (default: %(default)s)")
    parser.add_argument("--valid_batch_size",type=int,default=16,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size (default: %(default)s)", )
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--freeze", action="store_true", help="Use cuda") #ddd
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--multiple_decoder", action="store_true", help="Use cuda")
    parser.add_argument("--multiple_encoder", action="store_true", help="Use cuda")


    parser.add_argument("-see","--shared_entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--lrp_prog", action="store_true", help="use common lrp for progressive")
    parser.add_argument("--ind_lrp", action="store_true", help="use common lrp for progressive")
    parser.add_argument("--continue_training", action="store_true", help="continue training of the checkpoint")
    parser.add_argument("--save_path", type=str, default="/scratch/ResDSIC/models/", help="Where to Save model")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--sampling_training", action="store_true", help="Save model to disk")
    parser.add_argument("--joiner_policy", type=str, default = "res",help="Path to a checkpoint") 
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, default = "none")#/scratch/ResDSIC/models/zero__multi__0.0035_0.05_cond_ind_two-levels_Trueconcatenation_False/_very_best.pth.tar
    parser.add_argument("--checkpoint_base", type=str, default = "none",help="Path to a checkpoint") #"/scratch/universal-dic/weights/q2/model.pth"
    parser.add_argument("--tester", action="store_true", help="use common lrp for progressive")
    parser.add_argument("--only_progressive", action="store_true", help="onlu progressive") #rrrrr
    parser.add_argument("--support_progressive_slices",default=2,type=int,help="support_progressive_slices",)
    args = parser.parse_args(argv)
    return args