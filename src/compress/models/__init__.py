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


from .stf import SymmetricalTransFormer
from .cnn import WACNN
from .scalable.scalable_cnn import ResWACNN
from .scalable.shared_entropy import ResWACNNSharedEntropy
from .scalable.independent_entropy import ResWACNNIndependentEntropy
from .scalable.conditional_independent import ResWACNNConditionalIndependentEntropy
from .scalable.progressive import ProgressiveWACNN,ProgressiveMaskedWACNN
from .scalable.progressive_res import ProgressiveResWACNN
from .scalable.CHP_res import ChannelProgresssiveWACNN
from .scalable.progressive_enc import ProgressiveEncWACNN
from .tcm.scalable import ResTCM
from .scalable.utils import initialize_model_from_pretrained
from .video.google import ScaleSpaceFlow
from .video.scalable_video import ScalableScaleSpaceFlow
from .video.scalable_res_video import ResScalableScaleSpaceFlow


models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    "restcm":ResTCM,
    "resWacnn":ResWACNN,
    "shared":ResWACNNSharedEntropy,
    "independent":ResWACNNIndependentEntropy,
    "cond_ind":ResWACNNConditionalIndependentEntropy,
    "progressive": ProgressiveWACNN,
    "progressive_mask": ProgressiveMaskedWACNN,
    "progressive_res":ProgressiveResWACNN,
    "channel":ChannelProgresssiveWACNN,
    "progressive_enc":ProgressiveEncWACNN
}


video_models = {"ssf2020":ScaleSpaceFlow, 
                "full":ScalableScaleSpaceFlow,
                "res":ResScalableScaleSpaceFlow}

def get_model(args,device, lmbda_list):

    if args.model == "restcm":
        net = models[args.model](N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy,
                                joiner_policy = args.joiner_policy,
                                support_progressive_slices =args.support_progressive_slices,
        )



    elif  args.model == "progressive":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list

                        ) 
    elif  args.model == "progressive_enc":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension, #dddd
                                lmbda_list = lmbda_list

                        )       
    elif args.model == "progressive_mask":

        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy
                        )  
    
    elif args.model == "channel":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                multiple_hyperprior = args.multiple_hyperprior,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy,
                                joiner_policy = args.joiner_policy,
                                support_progressive_slices =args.support_progressive_slices,
                                double_dim = args.double_dim,
                                shared_entropy_estimation = False
                        )  
    elif args.model == "progressive_res":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder, #ddd
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                shared_entropy_estimation = args.shared_entropy_estimation,
                                joiner_policy = args.joiner_policy

                        )       
       
    elif args.model == "shared":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = lmbda_list,
                                multiple_decoder = args.multiple_decoder
                                )
    elif args.model == "cond_ind":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = lmbda_list,
                                multiple_decoder = args.multiple_decoder,
                                joiner_policy = args.joiner_policy
                                )
    else:
        net = models[args.model]( N = args.N,
                                M = args.M,
                                )


    net = net.to(device)
    return net
