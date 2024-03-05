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
from .scalable.progressive import ProgressiveWACNN
from .scalable.progressive_res import ProgressiveResWACNN


models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    "resWacnn":ResWACNN,
    "shared":ResWACNNSharedEntropy,
    "independent":ResWACNNIndependentEntropy,
    "cond_ind":ResWACNNConditionalIndependentEntropy,
    "progressive": ProgressiveWACNN,
    "progressive_res":ProgressiveResWACNN
}




def get_model(args,device, lmbda_list):


    if  args.model == "progressive":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list

                        )       

    elif args.model == "progressive_res":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
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
                                lrp_prog = args.lrp_prog,
                                independent_lrp = args.ind_lrp,
                                multiple_decoder = args.multiple_decoder,
                                joiner_policy = args.joiner_policy
                                )
    else:
        net = models[args.model]( N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = lmbda_list,
                                lrp_prog = args.lrp_prog,
                                independent_lrp = args.ind_lrp,
                                multiple_decoder = args.multiple_decoder
                                )


    net = net.to(device)
    return net
