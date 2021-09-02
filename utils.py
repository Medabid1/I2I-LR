import torch
import torch.nn as nn
from torch.nn import init 
import numpy as np 
from PIL import Image
import torchvision
import cv2


def init_net(net, init_type, gpu_ids=0):
    
    assert torch.cuda.is_available() == True, 'Cuda is not Available'
    net.to(gpu_ids) 
    init_weights(net, init_type)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal' :
                init.normal_(m.weight.data, 0, init_gain)
            elif init_type == 'xavier' :
                init.xavier_normal_(m.weight.data, init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthognal':
                init.orthognal_(m.weight.data, init_gain)
            else :
                raise NotImplementedError('Init method is not implemented')
        if hasattr(m, 'bias') and m.bias is not None :
            init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)


def downsample(x, size=8):
    if x.size(3) == size :
        return x 
    h = x.size(3) // size
    return torch.nn.AvgPool2d(h)(x)

def upsample(x, scale_factor=2):
    if scale_factor <= 1 :
        return x 
    return torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)(x)

def straight_through_estimator(x, r=127.5/4):
    xr = torch.round(x*r)/r
    return (xr - x).detach() + x

def downsample_mse(fake, x_low, margin=1):
    diff = torch.relu(torch.abs(downsample(fake, x_low.size(3)) - x_low).mean(dim=[1, 2, 3]) - margin)
    return diff.mean()

class space_to_channels(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.n = n 
    
    def forward(self, x):
        b, c , h, w = x.size()

        x = torch.reshape(x ,(b, c, h//self.n, self.n, w//self.n, self.n))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = torch.reshape(x,(b, c*(self.n**2), h//self.n, w//self.n))

        return x



def read_Image(path, size=256):
    x = Image.open(path)
    #resize = torchvision.transforms.RandomResizedCrop(size, scale=[0.8, 1.0], ratio=[0.9, 1.1]) 
    x = torchvision.transforms.functional.resize(x, size=(size,size))
    
    x = torchvision.transforms.functional.to_tensor(x)
    if x.size(0) > 3 :
        x = x[:3, :, :]
    x = torchvision.transforms.functional.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return x 

def color_change(src_path, ref_path,  size=256):
    from skimage import color
    from matplotlib import cm
    x_src = Image.open(src_path)
    lab_src = x_src
    ##lab_src = color.rgb2lab(x_src)
    x_ref = Image.open(ref_path)
    lab_ref = x_ref

    #lab_ref = color.rgb2lab(x_ref)

    src_mean =  np.mean(lab_src, axis=(0, 1), keepdims=True)
    src_var =  np.var(lab_src, axis=(0, 1), keepdims=True)
    ref_mean =  np.mean(lab_ref, axis=(0, 1), keepdims=True)
    ref_var =  np.var(lab_ref, axis=(0, 1), keepdims=True)
    
    x_target = ((lab_ref - ref_mean) / ref_var) * src_var + src_mean
    #x_target = color.lab2rgb(lab_ref_changed)
    x_target = Image.fromarray(np.uint8(x_target)).convert('RGB')
    x_src = Image.fromarray(np.uint8(x_src)).convert('RGB')

    
    x_target = torchvision.transforms.functional.resize(x_target, size=(size,size))
    x_target = torchvision.transforms.functional.to_tensor(x_target)
    print(x_target.size())
    if x_target.size(0) > 3 :
        x_target = x[:3, :, :]
    x_target = torchvision.transforms.functional.normalize(x_target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x_src = torchvision.transforms.functional.resize(x_src, size=(size,size))

    
    x_src = torchvision.transforms.functional.to_tensor(x_src)
    if x_target.size(0) > 3 :
        x_target = x[:3, :, :]
    x_src = torchvision.transforms.functional.normalize(x_src, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return x_src, x_target 

def downsample_eff(x, size=8, efficient_size=None ):
    if x.size(3) == size :
        return x 

    h = x.size(3) // size
    out = torch.nn.AvgPool2d(h)(x)
    if efficient_size :
        out2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(out)
        return out, out2 
    return out 


def read_file(sn,tn):
	s = cv2.imread(sn)
	s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
	t = cv2.imread(tn)
	t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
	return s, t

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def color_transfer(target, source):

    s, t = read_file(source, target)
    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    height, width, channel = s.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = s[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                s[i,j,k] = x

    s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
    t = cv2.cvtColor(t,cv2.COLOR_LAB2BGR)
    s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB) 

    s = Image.fromarray(s)
    t = Image.fromarray(t)
    x_target = torchvision.transforms.functional.resize(s, size=(128,128))
    x_target = torchvision.transforms.functional.to_tensor(x_target)
    x_target = torchvision.transforms.functional.normalize(x_target, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x_source = torchvision.transforms.functional.resize(t, size=(128,128))
    x_source = torchvision.transforms.functional.to_tensor(x_source)
    x_source = torchvision.transforms.functional.normalize(x_source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return x_source, x_target