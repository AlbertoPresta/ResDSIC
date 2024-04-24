
import torch.nn as nn
from compressai.layers import GDN, Win_noShift_Attention



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
            conv(N, M, kernel_size=1, stride=1), #32 
            GDN(M),
        )
        self.post_deocder = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),

        )

        

    def forward(self, x):
        out = x
        out_y = self.post_encoder(x)
        out = self.post_decoder(out_y)
        out += x

        return {
            "x_hat":out,
        }

