
import torch.nn as nn
from compress.layers import GDN, Win_noShift_Attention



def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class PostNet(nn.Module):

    def __init__(self, N = 96,M = 128, **kwargs):
        super().__init__( **kwargs)
        
        self.post_encoder = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), # halve 128
            GDN(N),
            conv(N, N, kernel_size=5, stride=2), # halve 64
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
            conv(N, N, kernel_size=5, stride=2), #32 
            GDN(N),
            conv(N, M, kernel_size=5, stride=2), # 16
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.post_decoder = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )


        

    def forward(self, x, residual):
        
        if residual is False:
            out_y = self.post_encoder(x)
            out = self.post_decoder(out_y)
            return out
        else:
            out = x 
            out = self.post_encoder(out)
            out = self.post_decoder(out) 
            out += x 
            return out          
