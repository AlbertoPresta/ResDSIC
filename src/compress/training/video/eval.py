
import argparse
import json
import math
import struct
import sys
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim
from torch import Tensor
from torch.cuda import amp
from torch.utils.model_zoo import tqdm

import compressai


from compress.datasets import RawVideoSequence, VideoFormat
from compressai.ops import compute_padding
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)
from compressai.zoo import video_models as pretrained_models


from compress.models import video_models
#models = {"ssf2020": ScaleSpaceFlow}


from .utils import (compute_metrics_for_frame,
                    write_body,
                    write_uints,
                    write_uchars, 
                    pad,
                    crop,
                    convert_yuv420_to_rgb,
                    estimate_bits_frame,

                    filesize,
                    aggregate_results)

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

RAWVIDEO_EXTENSIONS = (".yuv",)  # read raw yuv videos for now






@torch.no_grad()
def eval_model(net, sequence, binpath, keep_binaries = False):
    print("entro qua!!!")
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)

    f = binpath.open("wb")

    print(f" encoding {sequence.stem}", file=sys.stderr)
    # write original image size
    write_uints(f, (org_seq.height, org_seq.width))
    # write original bitdepth
    write_uchars(f, (org_seq.bitdepth,))
    # write number of coded frames
    write_uints(f, (num_frames,))
    with tqdm(total=num_frames) as pbar:
        print("the total number of frames is: ",num_frames)
        for i in range(num_frames):
            if i%200==0:
                print("current frame is: ",i)
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i == 0:
                x_rec, enc_info = net.encode_keyframe(x_cur)
                write_body(f, enc_info["shape"], enc_info["strings"])
                size = x_rec.size()
                num_pixels = size[0] * size[2] * size[3]
                # x_rec = net.decode_keyframe(enc_info["strings"], enc_info["shape"])
            else:
                x_rec, enc_info = net.encode_inter(x_cur, x_rec)
                for shape, out in zip(
                    enc_info["shape"].items(), enc_info["strings"].items()
                ):
                    write_body(f, shape[1], out[1])
                # x_rec = net.decode_inter(x_rec, enc_info["strings"], enc_info["shape"])

            x_rec = x_rec.clamp(0, 1)
            metrics = compute_metrics_for_frame(
                org_seq[i],
                crop(x_rec, padding),
                device,
                max_val,
            )

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    f.close()

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }

    seq_results["bitrate"] = (float(filesize(binpath)) * 8 * org_seq.framerate / (num_frames * 1000))

    

    seq_results["bpp"] = seq_results["bitrate"]/num_pixels
    print("num pixels is: ",num_pixels)
    if not keep_binaries:
        binpath.unlink()

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    
    print("SEQUENTIAL RESULTS: ",seq_results)
    return seq_results





@torch.no_grad()
def eval_model_scalable(net, sequence, binpath,mask_pol, quality, keep_binaries = False):
    print("entro qua!!!")
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)

    f = binpath.open("wb")

    print(f" encoding {sequence.stem}", file=sys.stderr)
    # write original image size
    write_uints(f, (org_seq.height, org_seq.width))
    # write original bitdepth
    write_uchars(f, (org_seq.bitdepth,))
    # write number of coded frames
    write_uints(f, (num_frames,))
    with tqdm(total=num_frames) as pbar:
        print("the total number of frames is: ",num_frames)
        for i in range(num_frames):
            if i%200==0:
                print("current frame is: ",i)
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i == 0:
                x_rec, enc_info = net.encode_keyframe(x_cur, quality, mask_pol)
                write_body(f, enc_info["shape"], enc_info["strings"])
                size = x_rec.size()
                num_pixels = size[0] * size[2] * size[3]
                # x_rec = net.decode_keyframe(enc_info["strings"], enc_info["shape"])
            else:
                x_rec, enc_info = net.encode_inter(x_cur, x_rec)
                for shape, out in zip(enc_info["shape"].items(), enc_info["strings"].items()):
                    write_body(f, shape[1], out[1])
                # x_rec = net.decode_inter(x_rec, enc_info["strings"], enc_info["shape"])

            if net.motion_input == "UMSR":
                x_rec_p = x_rec[1].clamp(0, 1)
                x_rec = x_rec[0].clamp(0, 1)
                metrics = compute_metrics_for_frame(
                    org_seq[i],
                    crop(x_rec_p, padding),
                    device,
                    max_val,
                )
            else:
                x_rec = x_rec.clamp(0, 1)
                metrics = compute_metrics_for_frame(
                    org_seq[i],
                    crop(x_rec, padding),
                    device,
                    max_val,
                )                

            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)
    f.close()

    seq_results: Dict[str, Any] = {k + "_" + str(quality): torch.mean(torch.stack(v)) for k, v in results.items()}

    seq_results["bitrate_" + str(quality)] = (float(filesize(binpath)) * 8 * org_seq.framerate / (num_frames * 1000))

    

    seq_results["bpp_" + str(quality)] = seq_results["bitrate"]/num_pixels
    print("num pixels is: ",num_pixels)
    if not keep_binaries:
        binpath.unlink()

    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    
    print("SEQUENTIAL RESULTS: ",seq_results)
    return seq_results





@torch.no_grad()
def eval_model_entropy_estimation(net: nn.Module, sequence: Path) -> Dict[str, Any]:
    org_seq = RawVideoSequence.from_file(str(sequence))

    if org_seq.format != VideoFormat.YUV420:
        raise NotImplementedError(f"Unsupported video format: {org_seq.format}")

    device = next(net.parameters()).device
    num_frames = len(org_seq)
    max_val = 2**org_seq.bitdepth - 1
    results = defaultdict(list)
    print(f" encoding {sequence.stem}", file=sys.stderr)
    with tqdm(total=num_frames) as pbar:
        for i in range(num_frames):
            x_cur = convert_yuv420_to_rgb(org_seq[i], device, max_val)
            x_cur, padding = pad(x_cur)

            if i == 0:
                x_rec, likelihoods = net.forward_keyframe(x_cur)  # type:ignore
            else:
                x_rec, likelihoods = net.forward_inter(x_cur, x_rec)  # type:ignore

            x_rec = x_rec.clamp(0, 1)

            metrics = compute_metrics_for_frame(
                org_seq[i],
                crop(x_rec, padding),
                device,
                max_val,
            )
            metrics["bitrate"] = estimate_bits_frame(likelihoods)

            print("the final metrics are: ",metrics)
            for k, v in metrics.items():
                results[k].append(v)
            pbar.update(1)

    seq_results: Dict[str, Any] = {
        k: torch.mean(torch.stack(v)) for k, v in results.items()
    }
    seq_results["bitrate"] = float(seq_results["bitrate"]) * org_seq.framerate / 1000
    for k, v in seq_results.items():
        if isinstance(v, torch.Tensor):
            seq_results[k] = v.item()
    return seq_results


def run_inference(
    filepaths,
    inputdir: Path,
    net: nn.Module,
    outputdir: Path,
    force: bool = False,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    **args: Any,
) -> Dict[str, Any]:
    results_paths = []

    for filepath in filepaths:
        print("filepath: ",filepath)
        output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(inputdir)
        output_subdir.mkdir(parents=True, exist_ok=True)
        sequence_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
        results_paths.append(sequence_metrics_path)

        if force:
            sequence_metrics_path.unlink(missing_ok=True)
        if sequence_metrics_path.is_file():
            continue

        with amp.autocast(enabled=args["half"]):
            with torch.no_grad():
                if entropy_estimation:
                    metrics = eval_model_entropy_estimation(net, filepath)
                else:
                    sequence_bin = sequence_metrics_path.with_suffix(".bin")
                    metrics = eval_model(
                        net, filepath, sequence_bin, args["keep_binaries"]
                    )
        with sequence_metrics_path.open("wb") as f:
            output = {
                "source": filepath.stem,
                "name": args["architecture"],
                "description": f"Inference ({description})",
                "results": metrics,
            }
            f.write(json.dumps(output, indent=2).encode())
    results = aggregate_results(results_paths)
    return results


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]

    net = video_models[arch]()
    net.update(force = True)
    net.load_state_dict(state_dict)
    if not no_update:
        net.update(force=True)
    net.eval()
    return net


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()







def create_parser():

    parser = argparse.ArgumentParser(
        description="Evaluate a video compression network on a video dataset.",
    )
    #parent_parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", 
                               type=str, 
                               default="/scratch/dataset/uvg/videos" 
                               ,help="sequences directory")
    parser.add_argument("--output", 
                               type=str,
                               default="/scratch/video_output" ,
                               help="output directory")
    parser.add_argument("-a","--architecture",
        type=str,
        choices=video_models.keys(),
        help="model architecture",
        default="ssf2020",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite previous runs"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda") #sss
    parser.add_argument("--half", action="store_true", help="use AMP")
    parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--keep_binaries",
        action="store_true",
        help="keep bitstream files in output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        choices=["pretrained", "checkpoint"],
        default="checkpoint",
        help="metric trained against (default: %(default)s)",
    )

    parser.add_argument(
        "-pth",
        "--path",
        type=str,
        
        default="/scratch/ssf_models",
        help="metric trained against (default: %(default)s)",
    )
    parser.add_argument(
        "-out_pth",
        "--output_path",
        type=str,
        default="",
        help="metric trained against (default: %(default)s)",
    )

    # Options for pretrained models
    #pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    parser.add_argument(
        "-q",
        "--quality",
        dest="quality",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )
   
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )
    return parser


import os
def create_runs(path, qualities):
    print("heeeeeyyy")
    f = [os.path.join(path,"q" + str(qualities[i]),"model.pth") for i in range(len(qualities))]
    return f



