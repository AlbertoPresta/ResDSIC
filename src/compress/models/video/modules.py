
import torch.nn as nn
from ..utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import QReLU
from compressai.ops import quantize_ste
from ..base import CompressionModel
from compress.layers.mask_layers import Mask

class Encoder(nn.Sequential):
    def __init__(self, in_planes: int, mid_planes: int = 128, out_planes: int = 192, factor:int = 1):
        super().__init__(
            conv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, out_planes*factor, kernel_size=5, stride=2),
        )

class Decoder(nn.Sequential):
    def __init__(self, out_planes: int, in_planes: int = 192, mid_planes: int = 128,factor:int = 1):
        super().__init__(
            deconv(in_planes*factor, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, out_planes, kernel_size=5, stride=2),
        )

class HyperEncoder(nn.Sequential):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192):
        super().__init__(
            conv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(mid_planes, mid_planes, kernel_size=5, stride=2),
        )

class HyperDecoder(nn.Sequential):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192):
        super().__init__(
            deconv(in_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(mid_planes, out_planes, kernel_size=5, stride=2),
        )

class HyperDecoderWithQReLU(nn.Module):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192):
        super().__init__()

        def qrelu(input, bit_depth=8, beta=100):
            return QReLU.apply(input, bit_depth, beta)

        self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu1 = qrelu
        self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
        self.qrelu2 = qrelu
        self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
        self.qrelu3 = qrelu

    def forward(self, x):
        x = self.qrelu1(self.deconv1(x))
        x = self.qrelu2(self.deconv2(x))
        x = self.qrelu3(self.deconv3(x))
        return x

class Hyperprior(CompressionModel):
    def __init__(self, planes: int = 192, mid_planes: int = 192,factor:int = 2):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(planes)
        self.hyper_encoder = HyperEncoder(planes*factor, mid_planes, planes)
        self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes*factor)
        self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes*factor)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.planes = planes

    def forward(self, y):
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)
        # mettere la maschera qua!!!!
        _, y_likelihoods = self.gaussian_conditional(y, scales, means)
        y_hat = quantize_ste(y - means) + means
        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)

        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_string = self.gaussian_conditional.compress(y, indexes, means)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        return y_hat
    


class HyperpriorMasked(Hyperprior):
    def __init__(self, 
                 planes: int = 192,
                 mid_planes: int = 192,
                 factor:int = 2, 
                 mask_policy = "two-levels", 
                 scalable_levels = 2):
        super().__init__(planes = planes, mid_planes=mid_planes,factor=factor)
        
        self.mask_policy = mask_policy
        self.scalable_levels = scalable_levels
        self.masking = Mask(self.mask_policy,scalable_levels = self.scalable_levels)
    

    def forward(self, y, quality, mask_pol = None, training = True,y_b = None):
        
        

        mask_pol  = self.mask_policy if mask_pol is None else mask_pol
        
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training = training)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)
        # mettere la maschera qua!!!!
        
        if quality == 0: # non maschero! 
            _, y_likelihoods = self.gaussian_conditional(y, scales, means)
            y_hat = quantize_ste(y - means) + means
        return y_hat, z_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y):
        z = self.hyper_encoder(y)

        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)

        indexes = self.gaussian_conditional.build_indexes(scales)
        y_string = self.gaussian_conditional.compress(y, indexes, means)
        y_hat = self.gaussian_conditional.quantize(y, "dequantize", means)

        return y_hat, {"strings": [y_string, z_string], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        scales = self.hyper_decoder_scale(z_hat)
        means = self.hyper_decoder_mean(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means)

        return y_hat   
    
    

    