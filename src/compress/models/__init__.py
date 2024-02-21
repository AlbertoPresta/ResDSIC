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
from .WACNN.scalable.single_decoder import scalable_icd 
from .WACNN.scalable.multiple_decoder import scalable_imd
from .WACNN.scalable.conditional_single_decoder import conditional_scalable_icd
from .WACNN.scalable.conditional_multiple_decoder import conditional_scalable_imd
from .WACNN.scalable.independent import ResWACNNIndependentEntropy

models = {
    "cimd":conditional_scalable_imd,
    "cicd":conditional_scalable_icd,
    "icd":scalable_icd,
    "imd":scalable_imd,
    'cnn': WACNN,
    "ind": ResWACNNIndependentEntropy
}



def configure_model(args):

    if args.model == "ind":
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lambda_list = args.lambda_list,
                                lrp_prog = args.lrp_prog,
                                independent_lrp = args.independent_lrp,
                                multiple_decoder =  args.multiple_decoder
                                )      

    if args.model == "cicd" or args.model == "cimd":
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lambda_list = args.lambda_list,
                                joiner_policy = args.joiner_policy
                                )   
    elif args.model == "imd" or args.model == "icd":
        net = models[args.model](N = args.N,
                                M = args.M,
                                mask_policy = args.mask_policy,
                                lambda_list = args.lambda_list,
                                )    

    else:
        net = models[args.model](N = args.N,
                                M = args.M,
                                )    
    return net