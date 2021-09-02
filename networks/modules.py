
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std
    
# In the Decoder
# one can call MS(x, mean, std)
# with the mean and std are from a PONO in the encoder
def MS_func(x, beta, gamma):
    return x * gamma + beta

class MS_mutli(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.conv = nn.Sequential(SnConv2d(2, 64, 3, 1, 1), 
                                  SnConv2d(64, dim_out*2, 3, 1, 1))

    def forward(self, x, beta=None, gamma=None):
        inputs = torch.cat((beta, gamma), dim=1)
        h  = self.conv(inputs)
        mu, std = torch.split(h, self.dim_out, dim=1)
        x = x.mul(std) + mu
        return x
        
class SnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
    def forward(self, x):
        return self.conv(x)

class MS(nn.Module):
    def __init__(self, dim_in):
        super(MS, self).__init__()
        self.dim_in = dim_in
        self.conv = SnConv2d(2, 2, 3, 1, 1)
        #self.conv_color = SnConv2d(1, 3, 3, 1, 1)

    def forward(self, x, beta=None, gamma=None):
        inputs = torch.cat((beta, gamma), dim=1)
        h  = self.conv(inputs)
        mu, std = torch.split(h, 1, dim=1)
        #x = self.conv_color(x)
        x = x.mul(std) + mu
        return x

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False, use_pono=False):
        super().__init__()
        assert not (downsample and upsample), 'must choose either down or up'
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.upsample = upsample
        self.use_pono = use_pono
        if use_pono :
            self.pono = PONO
            
        self.k = 0
        if upsample :
            self.k = 3
        
        self._build_weights(dim_in, dim_out)


    def _build_weights(self, dim_in, dim_out):
        self.conv1 = SnConv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = SnConv2d(dim_out, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        if self.learned_sc:
            self.conv1x1 = SnConv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        return x

    def _residual(self, x):
        self.stats = []
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.use_pono :
            x, mu_1, std_1 = self.pono(x)
            self.stats.append((mu_1, std_1))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        if self.use_pono :
            x, mu_2, std_2 = self.pono(x)
            self.stats.append((mu_2, std_2))
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        if self.use_pono :
            return x / math.sqrt(2), self.stats
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.utils.spectral_norm(nn.Linear(style_dim, num_features*2))

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = SnConv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = SnConv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = SnConv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class SPAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN,self).__init__()
        self.conv_weight = nn.Conv2d(input_nc, planes, 1)
        self.conv_bias = nn.Conv2d(input_nc, planes, 1)
        self.norm = norm(planes)
    
    def forward(self,x,addition):

        x = self.norm(x)
        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out


class SPAdaInResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), 
            downsample=False, upsample=False, use_ms=False):
        super().__init__()
        assert not (downsample and upsample), 'must choose either down or up'
        self.use_ms = use_ms
        self.actv = actv
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.upsample = upsample
        if use_ms :
            self.ms = MS

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = SnConv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = SnConv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = SPAdaIN(nn.InstanceNorm2d, 3, dim_in)
        self.norm2 = SPAdaIN(nn.InstanceNorm2d, 3, dim_out)
        if self.use_ms :
            self.ms1 = MS(dim_out)
            self.ms2 = MS(dim_out)

        if self.learned_sc:
            self.conv1x1 = SnConv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x, lr):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
        return x

    def _residual(self, x, lr, stats=None):
        
        x = self.norm1(x, lr)
        x = self.actv(x)
        x = self.conv1(x)
        if self.use_ms :
            mu_2, std_2 = stats[1]
            x = self.ms1(x, mu_2, std_2)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.upsample:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            lr = nn.UpsamplingBilinear2d(scale_factor=2)(lr)
        x = self.norm2(x, lr)
        x = self.actv(x)
        x = self.conv2(x)
        if self.use_ms :
            mu_1, std_1 = stats[0]
            x = self.ms2(x, mu_1, std_1)
        return x

    def forward(self, x, lr, stats=None):
        x = self._shortcut(x, lr) + self._residual(x, lr, stats)
        return x / math.sqrt(2)  # unit variance