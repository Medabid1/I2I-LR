import torch 
import torch.nn as nn 

from utils import straight_through_estimator, downsample

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")

    def set_alpha(self, alpha, stage):
        self.alpha = alpha 
        self.stage = stage 

class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps):
        # Obtain predictions
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps):
        # Obtain predictions

        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(torch.nn.ReLU()(1 + r_f_diff))
                + torch.mean(torch.nn.ReLU()(1 - f_r_diff)))

class R1Loss(GANLoss):
    def __init__(self, dis, gamma=10):
        self.dis = dis
        self.gamma = gamma 
    def r1loss(self, inputs, label=None):
    # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return torch.nn.functional.softplus(l*inputs).mean()
    
    def dis_loss(self, real_samps, fake_samps):
        real_samps.requires_grad = True 

        r_preds = self.dis(real_samps)
        d_real_loss = self.r1loss(r_preds, True)

        grad_real = torch.autograd.grad(outputs=d_real_loss.sum(), inputs=real_samps, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1)**2).mean()
        grad_penalty = 0.5*self.gamma*grad_penalty

        d_r_loss = d_real_loss + grad_penalty
        
        f_preds = self.dis(fake_samps.detach())
        d_f_loss = self.r1loss(f_preds, label=False)

        return d_r_loss + d_f_loss
    
    def gen_loss(self, fake_samps):
        f_preds = self.dis(fake_samps.detach())
        d_f_loss = self.r1loss(f_preds, label=True)
        return d_f_loss



class R1Loss_STE(GANLoss):
    def __init__(self, dis, gamma=10):
        self.dis = dis
        self.gamma = gamma 
    
    def r1loss(self, inputs, label=None):
    # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return torch.nn.functional.softplus(l*inputs).mean()
     
    def dis_loss(self, real_samps, real_low, fake_samps):
        self.dis.train()
        real_samps.requires_grad = True 

        r_diff = straight_through_estimator(torch.abs(real_low - real_low))
        f_diff = straight_through_estimator(torch.abs(downsample(fake_samps, real_low.size(3)) - real_low))
        
        r_preds = self.dis(real_samps, r_diff)
        d_real_loss = self.r1loss(r_preds, True)

        grad_real = torch.autograd.grad(outputs=d_real_loss.sum(), inputs= real_samps, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1)**2).mean()

        grad_penalty =  0.5*self.gamma*grad_penalty
        
        d_r_loss = d_real_loss + grad_penalty 
        
        f_preds = self.dis(fake_samps, f_diff)
        d_f_loss = self.r1loss(f_preds, label=False)
        return d_r_loss + d_f_loss
    
    def gen_loss(self, fake_samps, real_low):
        self.dis.eval()
        f_diff = straight_through_estimator(torch.abs(downsample(fake_samps, real_low.size(3)) - real_low))
        f_preds = self.dis(fake_samps, f_diff)
        d_f_loss = self.r1loss(f_preds, label=True)
        return d_f_loss

class R1Loss_color(GANLoss):
    def __init__(self, dis, gamma=10):
        self.dis = dis
        self.gamma = gamma 
    
    def r1loss(self, inputs, label=None):
    # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return torch.nn.functional.softplus(l*inputs).mean()
     
    def dis_loss(self, real_samps, real_low, fake_samps):
        self.dis.train()
        real_samps.requires_grad = True 

        fake_low = downsample(fake_samps, real_low.size(3))
        fake_low_grey = (fake_low[:, 0, :, :] * 0.3 + fake_low[:, 1, :, :] * 0.59 +  fake_low[:, 2, :, :] * 0.11).unsqueeze(1)        
        
        r_diff = straight_through_estimator(torch.abs(real_low - real_low))
        f_diff = straight_through_estimator(torch.abs(fake_low_grey  - real_low))
        
        r_preds = self.dis(real_samps, r_diff)
        d_real_loss = self.r1loss(r_preds, True)

        grad_real = torch.autograd.grad(outputs=d_real_loss.sum(), inputs= real_samps, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1)**2).mean()

        grad_penalty =  0.5*self.gamma*grad_penalty
        
        d_r_loss = d_real_loss + grad_penalty 
        
        f_preds = self.dis(fake_samps, f_diff)
        d_f_loss = self.r1loss(f_preds, label=False)
        return d_r_loss + d_f_loss
    
    def gen_loss(self, fake_samps, real_low):
        self.dis.eval()
        fake_low = downsample(fake_samps, real_low.size(3))
        fake_low_grey = (fake_low[:, 0, :, :] * 0.3 + fake_low[:, 1, :, :] * 0.59 +  fake_low[:, 2, :, :] * 0.11).unsqueeze(1)
        f_diff = straight_through_estimator(torch.abs(fake_low_grey - real_low))
        
        f_preds = self.dis(fake_samps, f_diff)
        d_f_loss = self.r1loss(f_preds, label=True)
        return d_f_loss