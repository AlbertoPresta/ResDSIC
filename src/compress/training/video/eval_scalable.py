

import json
import sys
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.model_zoo import tqdm
from compress.datasets import RawVideoSequence, VideoFormat
from compress.models import video_models

from .utils import (compute_metrics_for_frame,
                    write_body,
                    write_uints,
                    write_uchars, 
                    pad,
                    crop,
                    convert_yuv420_to_rgb,
                    filesize,
                    aggregate_results)


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


def run_inference_scalable(
    filepaths,
    inputdir: Path,
    net: nn.Module,
    outputdir: Path,
    mask_pol: str,
    list_pr: list = [0,1,2,3,4,5,6,7,8,9,10],
    force: bool = False,
    trained_net: str = "",
    description: str = "",
    write_results = False,
    **args: Any,
) -> Dict[str, Any]:
    

    results_across_quality = {}

    for q in list_pr:
        results_paths = []
        for filepath in filepaths:
            print("filepath: ",filepath)

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(inputdir)
            output_subdir.mkdir(parents=True, exist_ok=True)
            sequence_metrics_path = output_subdir / f"{filepath.stem}-{trained_net}.json"
            #results_paths.append(sequence_metrics_path)

            if force:
                sequence_metrics_path.unlink(missing_ok=True)
            if sequence_metrics_path.is_file():
                continue

            with amp.autocast(enabled=args["half"]):
                with torch.no_grad():
                    sequence_bin = sequence_metrics_path.with_suffix(".bin")
                    metrics = eval_model_scalable(net, 
                                                  filepath, 
                                                  sequence_bin,
                                                  mask_pol,
                                                  q, 
                                                  args["keep_binaries"])
            with sequence_metrics_path.open("wb") as f:
                output = {
                    "source": filepath.stem,
                    "name": args["architecture"],
                    "description": f"Inference ({description})",
                    "results": metrics,
                }
                f.write(json.dumps(output, indent=2).encode())
        results = aggregate_results(results_paths)
        results_across_quality[str(q)] = results
    return results_across_quality