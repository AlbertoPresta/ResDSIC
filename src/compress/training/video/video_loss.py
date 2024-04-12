
import math
from collections import defaultdict
import torch
import torch.nn as nn


def collect_likelihoods_list(likelihoods_list, num_pixels: int):
    bpp_info_dict = defaultdict(int)
    bpp_loss = 0

    for i, frame_likelihoods in enumerate(likelihoods_list):
        #print("***************** ",i)
        frame_bpp = 0
        #print("tpye_ ",type(frame_likelihoods))
        for label, likelihoods in frame_likelihoods.items():
            #print("label: ",label)
            label_bpp = 0
            for field, v in likelihoods.items(): # hyperprior + main

                bpp = torch.log(v).sum(dim=(1, 2, 3)) / (-math.log(2) * num_pixels)

                bpp_loss += bpp
                frame_bpp += bpp
                label_bpp += bpp

                bpp_info_dict[f"bpp_loss.{label}"] += bpp.sum()
                bpp_info_dict[f"bpp_loss.{label}.{field}"] += bpp.sum() #dddd
                bpp_info_dict[f"bpp_loss.{label}.{i}.{field}"] = bpp.sum()
            bpp_info_dict[f"bpp_loss.{label}.{i}"] = label_bpp.sum()
        bpp_info_dict[f"bpp_loss.{i}"] = frame_bpp.sum() #ddddd
    
    bpp_info_dict[f"bpp_loss_total"] = bpp_loss
    return bpp_loss, bpp_info_dict



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target):
        assert isinstance(target, type(output["x_hat"]))
        assert len(output["x_hat"]) == len(target)

        self._check_tensors_list(target)
        self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions = []
        distortions = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion, distortion = self._get_scaled_distortion(x_hat, x)

            distortions.append(distortion)
            scaled_distortions.append(scaled_distortion)

            if self.return_details:
                out[f"frame{i}.mse_loss"] = distortion
        # aggregate (over batch and frame dimensions).
        out["mse_loss"] = torch.stack(distortions).mean()

        # average scaled_distortions accros the frames
        scaled_distortions = sum(scaled_distortions) / num_frames

        assert isinstance(output["likelihoods"], list)
        likelihoods_list = output.pop("likelihoods")

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss, bpp_info_dict = collect_likelihoods_list(likelihoods_list, num_pixels)
        #out.update(bpp_info_dict)  # detailed bpp: per frame, per latent, etc...
        
        for k,v in bpp_info_dict.item():
            bpp_info_dict[k] = v.mean()
        out["bpp_info_dict"] = bpp_info_dict

        # now we either use a fixed lambda or try to balance between 2 lambdas
        # based on a target bpp.
        lambdas = torch.full_like(bpp_loss, self.lmbda)


        out["bpp_loss"] = bpp_loss
        bpp_loss = bpp_loss.mean()
        out["bpp_loss_mean"] = bpp_loss
        out["loss"] = (lambdas * scaled_distortions).mean() + bpp_loss

        out["distortion"] = scaled_distortions.mean()
        out["bpp_loss"] = bpp_loss
        return out
    




class ScalableRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=[1e-2,0.3], return_details: bool = False, bitdepth: int = 8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda
        self._scaling_functions = lambda x: (2**bitdepth - 1) ** 2 * x
        self.return_details = bool(return_details)

    @staticmethod
    def _get_rate(likelihoods_list, num_pixels):
        return sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for frame_likelihoods in likelihoods_list
            for likelihoods in frame_likelihoods
        )

    def _get_scaled_distortion(self, x, target):
        if not len(x) == len(target):
            raise RuntimeError(f"len(x)={len(x)} != len(target)={len(target)})")

        nC = x.size(1)
        if not nC == target.size(1):
            raise RuntimeError(
                "number of channels mismatches while computing distortion"
            )

        if isinstance(x, torch.Tensor):
            x = x.chunk(x.size(1), dim=1)

        if isinstance(target, torch.Tensor):
            target = target.chunk(target.size(1), dim=1)

        # compute metric over each component (eg: y, u and v)
        metric_values = []
        for x0, x1 in zip(x, target):
            v = self.mse(x0.float(), x1.float())
            if v.ndimension() == 4:
                v = v.mean(dim=(1, 2, 3))
            metric_values.append(v)
        metric_values = torch.stack(metric_values)

        # sum value over the components dimension
        metric_value = torch.sum(metric_values.transpose(1, 0), dim=1) / nC
        scaled_metric = self._scaling_functions(metric_value)

        return scaled_metric, metric_value

    @staticmethod
    def _check_tensor(x) -> bool:
        return (isinstance(x, torch.Tensor) and x.ndimension() == 4) or (
            isinstance(x, (tuple, list)) and isinstance(x[0], torch.Tensor)
        )

    @classmethod
    def _check_tensors_list(cls, lst):
        if (
            not isinstance(lst, (tuple, list))
            or len(lst) < 1
            or any(not cls._check_tensor(x) for x in lst)
        ):
            raise ValueError(
                "Expected a list of 4D torch.Tensor (or tuples of) as input"
            )

    def forward(self, output, target):
        #assert isinstance(target, type(output["x_hat"])) #dddd
        #assert len(output["x_hat"]) == len(target)

        #self._check_tensors_list(target)
        #self._check_tensors_list(output["x_hat"])

        _, _, H, W = target[0].size()
        num_frames = len(target)
        out = {}
        num_pixels = H * W * num_frames

        # Get scaled and raw loss distortions for each frame
        scaled_distortions_base = []
        distortions_base = []
        scaled_distortions_prog = []
        distortions_prog = []
        for i, (x_hat, x) in enumerate(zip(output["x_hat"], target)):
            scaled_distortion_base, distortion_base = self._get_scaled_distortion(x_hat[0], x)

            distortions_base.append(distortion_base)
            scaled_distortions_base.append(scaled_distortion_base)

            scaled_distortion_prog, distortion_prog = self._get_scaled_distortion(x_hat[1], x)
            distortions_prog.append(distortion_prog)
            scaled_distortions_prog.append(scaled_distortion_prog)

        # aggregate (over batch and frame dimensions).
        out["mse_base"] = torch.stack(distortions_base).mean()
        out["mse_prog"] = torch.stack(distortions_prog).mean()
        out["mse_loss"] = out["mse_base"] + out["mse_prog"] 
        # average scaled_distortions accros the frames
        scaled_distortions_base = sum(scaled_distortions_base) / num_frames
        scaled_distortions_prog = sum(scaled_distortions_prog) / num_frames
        
        likelihoods_list_base = output["likelihoods_base"]
        likelihoods_list_prog = output["likelihoods_prog"]

        # collect bpp info on noisy tensors (estimated differentiable entropy)
        bpp_loss_base, bpp_info_dict_base = collect_likelihoods_list(likelihoods_list_base,
                                                                      num_pixels)

        bpp_loss_prog, bpp_info_dict_prog = collect_likelihoods_list(likelihoods_list_prog,
                                                                      num_pixels)


        for k,v in bpp_info_dict_base.items():
            bpp_info_dict_base[k] = v.mean()

        for k,v in bpp_info_dict_prog.items():
            bpp_info_dict_prog[k] = v.mean()
        
        out["bpp_info_dict"] = bpp_info_dict_base
        out["bpp_info_dict_prog"] = bpp_info_dict_prog


        #print("PRENDIAMO IL BASE:",bpp_loss_base)
        #print(bpp_info_dict_base)

        # now we either use a fixed lambda or try to balance between 2 lambdas #dddd
        # based on a target bpp.
        lambdas_base = torch.full_like(bpp_loss_base, self.lmbda[0])
        lambdas_prog = torch.full_like(bpp_loss_prog, self.lmbda[1])



        out["bpp_loss_totality"] = bpp_loss_base + bpp_loss_prog
        bpp_loss_base = bpp_loss_base.mean()
        bpp_loss_prog = bpp_loss_prog.mean()
        out["bpp_loss"] = bpp_loss_base + bpp_loss_prog
        out["bpp_base"] = bpp_loss_base
        out["bpp_prog"] = bpp_loss_prog
        out["bpp_total"] = bpp_loss_base + bpp_loss_prog
        
        out["loss_base"] = (lambdas_base * scaled_distortions_base).mean() + bpp_loss_base
        out["loss_prog"] = (lambdas_prog * scaled_distortions_prog).mean() + bpp_loss_prog
        out["loss"] = out["loss_prog"] + out["loss_base"]
        out["distortion_base"] = scaled_distortions_base.mean()
        out["distortion_prog"] = scaled_distortions_prog.mean()
        out["distortion"] = out["distortion_base"] + out["distortion_prog"] 

        return out