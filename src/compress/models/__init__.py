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



from .WACNN.cnn import WACNN
from .WACNN.scalable.scalable_cnn import ResWACNN
from .WACNN.scalable.shared_entropy import ResWACNNSharedEntropy
from .WACNN.scalable.independent_entropy import ResWACNNIndependentEntropy
from .WACNN.scalable.conditional_independent import ConditionalWACNN
from .WACNN.scalable.conditional_shared import ConditionalSharedWACNN


models = {
    
    'cnn': WACNN,
    "resWacnn":ResWACNN,
    "shared":ResWACNNSharedEntropy,
    "independent":ResWACNNIndependentEntropy,
    "conditional":ConditionalWACNN,
    "conditional_shared":ConditionalSharedWACNN
}



def configure_model(args):
    if args.model == "conditional":
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = args.lambda_list,
                                joiner_policy = args.joiner_policy
                                )

    
    elif args.model == "conditional_shared":
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = args.lambda_list,
                                joiner_policy = args.joiner_policy,
                                independent_hyperprior = args.independent_hyperprior
                                )
      
    elif args.model == "independent":
        net = models[args.model](N = args.N, 
                                 M = args.M,
                                 mask_policy = args.mask_policy,
                                 lambda_list = args.lambda_list,
                                 independent_latent_hyperprior = args.independent_latent_hyperprior,
                                 independent_blockwise_hyperprior = args.independent_blockwise_hyperprior)

    else:
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lmbda_list = args.lambda_list,
                                lrp_prog = args.lrp_prog,
                                independent_lrp = args.ind_lrp,
                                )
    return net