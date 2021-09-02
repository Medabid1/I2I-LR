import torch 
import torch.nn as nn 
import numpy as np 

from .modules import ResBlk, AdainResBlk, SnConv2d, SPAdaInResBlk, SPAdaInResBlk01
from utils import upsample





class Discriminator(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512, low_size=8):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [SnConv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - int(np.log2(low_size))
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)
        dim_out = min(dim_out*2, max_conv_dim)
        repeat_num = int(np.log2(low_size)) - 3
        
        last_block = [ResBlk(dim_in + 3, dim_out, downsample=False)]
        for i in range(repeat_num):
            last_block += [ResBlk(dim_out, dim_out, downsample=True)]
        
        last_block += [nn.LeakyReLU(0.2)]
        last_block += [SnConv2d(dim_out , dim_out, 4, 1, 0)]
        last_block += [nn.LeakyReLU(0.2)]
        last_block += [SnConv2d(dim_out, 1, 1, 1, 0)]
        self.last_block = nn.Sequential(*last_block)

    def forward(self, x, diff):
        out = self.main(x)
        out = torch.cat((out, diff), dim=1)
        out = self.last_block(out)
        return out

class Generator(nn.Module):
    def __init__(self, dim_in, n_res, img_size=128):
        super().__init__()
        n = int(np.log2(img_size)) - 3
        down_block = [SnConv2d(3, dim_in, 3, 1, 1)]
        self.n_up = n 
        for _ in range(n):
            down_block += [ResBlk(dim_in, dim_in, normalize=True, downsample=True)]
        self.down_block = nn.Sequential(*down_block)

        block = []
        dim_out = dim_in // 2 
        for _ in range(n):
            block.append(ResBlk(dim_in, dim_out, upsample=True))
            dim_in = dim_out
            dim_out = dim_out // 2

        self.block = nn.ModuleList(block)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2),
            SnConv2d(dim_in, 3, 1, 1, 0))

    def forward(self, x, lr):
        h = self.down_block(x)
        
        for i, block in enumerate(self.block):
            h0 = upsample(lr,  2**(i))
            h = block(h, h0)
        return self.to_rgb(h)


class AdaInGenerator(nn.Module):
    def __init__(self, dim_in, n_res, img_size=128, style_dim=64):
        super().__init__()
        
        k_in = 64 
        k_out = 128
        n = int(np.log2(img_size)) - 3
        down_block = [SnConv2d(3, k_in, 3, 1, 1)]
 
        for _ in range(n):
            down_block += [ResBlk(k_in, k_out, normalize=True, downsample=True)]
            k_in = k_out 
            k_out = min(2 * k_out, dim_in) 
        
        down_block += [ResBlk(k_out, k_out, normalize=True, downsample=False)]
        self.down_block = nn.Sequential(*down_block)
        block = [AdainResBlk(k_out, k_out, upsample=False, style_dim=style_dim)]
    
        k_in = k_out 
        k_out = k_out // 2
        for _ in range(n):
            block.append(AdainResBlk(k_in, k_out, upsample=True, style_dim=style_dim))
            k_in = k_out
            k_out = max(k_out // 2, 64)

        self.block = nn.ModuleList(block)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2),
            SnConv2d(k_out, 3, 1, 1, 0))

    def forward(self, x, s):
        h = self.down_block(x)
        for i, block in enumerate(self.block):
            h = block(h, s)
        return self.to_rgb(h)

class AdaInGenerator(nn.Module):
    def __init__(self, dim_in, n_res, img_size=128, style_dim=64):
        super().__init__()
        
        k_in = 64 
        k_out = 128
        n = int(np.log2(img_size)) - 3
        down_block = [SnConv2d(3, k_in, 3, 1, 1)]
 
        for _ in range(n):
            down_block += [ResBlk(k_in, k_out, normalize=True, downsample=True)]
            k_in = k_out 
            k_out = min(2 * k_out, dim_in) 
        
        down_block += [ResBlk(k_out, k_out, normalize=True, downsample=False)]
        self.down_block = nn.Sequential(*down_block)
        block = [AdainResBlk(k_out, k_out, upsample=False, style_dim=style_dim)]
    
        k_in = k_out 
        k_out = k_out // 2
        for _ in range(n):
            block.append(AdainResBlk(k_in, k_out, upsample=True, style_dim=style_dim))
            k_in = k_out
            k_out = max(k_out // 2, 64)

        self.block = nn.ModuleList(block)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2),
            SnConv2d(k_out, 3, 1, 1, 0))

    def forward(self, x, s):
        h = self.down_block(x)
        for i, block in enumerate(self.block):
            h = block(h, s)
        return self.to_rgb(h)

class SPadaInGenerator(nn.Module):
    def __init__(self, dim_in, n_res, img_size=128):
        super().__init__()
        
        k_in = 64 
        k_out = 128
        n = int(np.log2(img_size)) - 3
        down_block = [SnConv2d(3, k_in, 3, 1, 1)]
 
        for _ in range(n):
            down_block += [ResBlk(k_in, k_out, normalize=True, downsample=True)]
            k_in = k_out 
            k_out = min(2 * k_out, dim_in) 
        
        down_block += [ResBlk(k_out, k_out, normalize=True, downsample=False)]
        self.down_block = nn.Sequential(*down_block)
        block = [SPAdaInResBlk(k_out, k_out, upsample=False)]
    
        k_in = k_out 
        k_out = k_out // 2
        for _ in range(n):
            block.append(SPAdaInResBlk(k_in, k_out, upsample=True))
            k_in = k_out
            k_out = max(k_out // 2, 64)

        self.block = nn.ModuleList(block)
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2),
            SnConv2d(k_out, 3, 1, 1, 0))

    def forward(self, x, lr):
        h = self.down_block(x)
        for i, block in enumerate(self.block):
            h0 = upsample(lr, 2**(i - 1))
            h = block(h, h0)
        return self.to_rgb(h)



class PonoSPadaInGenerator(nn.Module):
    def __init__(self, dim_in, img_size=128, low_size=8):
        super().__init__()
        
        k_in = 64 
        k_out = 128 
        n = int(np.log2(img_size)) - int(np.log2(low_size)) 
        self.from_rgb = nn.Sequential(SnConv2d(3, k_in, 3, 1, 1))
        down_block = []
        for _ in range(n):
            down_block += [ResBlk(k_in, k_out, normalize=True, downsample=True, use_pono=True)]
            k_in = k_out 
            k_out = min(2 * k_out, dim_in) 
        
        down_block += [ResBlk(k_out, k_out, normalize=True, downsample=False, use_pono=True)]
        self.down_block = nn.ModuleList(down_block)
        block = [SPAdaInResBlk(k_out, k_out, upsample=False, use_ms=True)]
    
        k_in = k_out 
        k_out = k_out // 2
        for _ in range(n):
            block.append(SPAdaInResBlk(k_in, k_out, upsample=True, use_ms=True))
            k_in = k_out
            k_out = max(k_out // 2, 64)

        self.block = nn.ModuleList(block)
        self.to_rgb = nn.Sequential(nn.InstanceNorm2d(k_out, affine=True),
            nn.LeakyReLU(0.2),
            SnConv2d(k_out, 3, 1, 1, 0))

    def forward(self, x, lr):
        h = self.from_rgb(x)
        stats = []
        for j, dblock in enumerate(self.down_block):
            h, stat = dblock(h) 
            stats.append(stat)
        stats = stats[::-1]
        for i, block in enumerate(self.block):
            h0 = upsample(lr, 2**(i - 1))
            h = block(h, h0, stats[i])
        return self.to_rgb(h)