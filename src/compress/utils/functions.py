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


import argparse
from compressai.zoo import models

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default='resWacnn18',
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda_starter",
        dest="lmbda_starter",
        type=float,
        default=0.05,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--scalable_levels", type=int, default=4, help="Batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--num_images", type=int, default=16000, help="Batch size (default: %(default)s)"
    )



    parser.add_argument(
        "--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--mask_policy", type=str, default="learnable-mask", help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args