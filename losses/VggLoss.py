from collections import namedtuple
import torch
import torchvision 
import torch.nn as nn

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        import os
        os.environ['TORCH_HOME'] = './logs'
        torch.hub.set_dir('./logs')
        self.vgg_feature = torchvision.models.vgg16(pretrained = True).features[:23].to('cuda')
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
                         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3']
        
        self.style_list = [ 'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.style_w = [ 0.03125, 0.0625, 0.125, 1.]

        self.content = 'relu4_3'
        for parameter in self.parameters():
            parameter.requires_grad = False
 
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to('cuda')
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to('cuda')
        self.resize = resize
             

    def forward(self, target, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std


        loss = 0.0
        x = input
        y = target
        w_i = 0
        for name, layer in zip(self.vgg_layer, self.seq_list) :
            x = layer(x)
            y = layer(y)
            if name in self.style_list :
                loss += self.style_w[w_i] * torch.nn.functional.l1_loss(x, y)   
                w_i += 1      
        return loss