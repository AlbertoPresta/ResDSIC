
from typing import Any,  List, Optional, Tuple, Union
import torch
from torch import Tensor
from compressai.entropy_models import  GaussianConditional



class GaussianConditionalMask(GaussianConditional):

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(scale_table = scale_table, *args, **kwargs)


    def quantize(
        self, inputs , mode, means = None, mask = None):
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        
        if mode == "noise":
            
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            if means is not None:
                inputs = inputs - means
            if mask is not None:
                inputs = inputs*mask
                noise = noise*mask # per ora lo lascio rumoroso!
            inputs = inputs + noise

            if means is not None: 
                inputs = inputs + means
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        if mask is not None:
            outputs = outputs*mask
        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs


    def _likelihood(self, inputs: Tensor, scales: Tensor, means = None, mask = None):
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs
        
        if mask is not None: 
            values = values*mask

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, inputs, scales, means = None, training = None, mask = None):
        if training is None:
            training = self.training

        if mask is not None: 
            scales = scales*mask


        outputs = self.quantize(inputs, "noise" if training else "dequantize", means, mask = mask)
        likelihood = self._likelihood(outputs, scales, means, mask = mask)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        return outputs, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
