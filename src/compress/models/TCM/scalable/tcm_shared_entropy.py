

from compress.entropy_models import EntropyBottleneck, GaussianConditional 
import torch.nn as nn 
from ..tcm import TCM
import torch 
from compressai.layers import ResidualBlockWithStride,ConvTransBlock,conv3x3

class SharedTCM(TCM):
    def __init__(self, config=[2, 2, 2, 2, 2, 2],
                head_dim=[8, 16, 32, 32, 16, 8],
                drop_path_rate=0, 
                N=128,  
                M=320, 
                num_slices=5, 
                max_support_slices=5,
                scalable_levels = 4,
                mask_policy = "learnable-mask",
                lmbda_list = [0.05],
                **kwargs):
        super().__init__(config=config,
                        head_dim=head_dim,
                        drop_path_rate=drop_path_rate, 
                        N=N, 
                        M=M, 
                        num_slices=num_slices,
                        max_support_slices=max_support_slices,
                        **kwargs   )
        
        assert lmbda_list is not None 

        self.mask_policy = mask_policy
        self.lmbda_list = lmbda_list 
        self.scalable_levels = scalable_levels 

        self.halve = 8
        self.level = 5 if self.halve == 8 else -1 
        self.factor = self.halve**2
        assert self.N%self.factor == 0 
        self.T = int(self.N//self.factor) + 3

        dim = N

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0


        self.scalable_levels = len(lmbda_list)
        self.lmbda_list = lmbda_list
        self.lmbda_index_list = dict(zip(self.lmbda_list, [i  for i in range(len(self.lmbda_list))] ))


        print(self.lmbda_index_list)
        print("questa è la lista finale",self.lmbda_index_list)
        
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.entropy_bottleneck_prog = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_prog = GaussianConditional(None)

        if self.mask_policy == "learnable-mask":
            self.gamma = torch.nn.Parameter(torch.ones((self.scalable_levels - 1, self.M))) #il primo e il base layer
            self.mask_conv = nn.Sequential(torch.nn.Conv2d(in_channels=self.M, out_channels=self.M, kernel_size=1, stride=1),)




        self.m_down1_progressive = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2_progressive = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3_progressive = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        
        self.g_a_progressive = nn.Sequential(*[ResidualBlockWithStride(self.T, 2*N, 2)] + \
                                        self.m_down1_progressive + \
                                          self.m_down2_progressive + \
                                            self.m_down3_progressive)


    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """

        aux_loss1 = self.entropy_bottleneck_prog.loss()
        aux_loss2 = self.entropy_bottleneck.loss()
        aux_loss = aux_loss1 + aux_loss2 

        #aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss
