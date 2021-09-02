import torch 
import torch.nn as nn 
import torchvision
from itertools import chain, combinations
from networks.networks import PonoSPadaInGenerator, Discriminator
from losses.GanLoss import R1Loss_STE
from logger import WandbLogger
from utils import downsample, init_net, downsample_mse, read_Image
#import wandb


class Model:

    def __init__(self, img_size, low_size, dim_in, epochs, checkpoint_path, device, use_ema):
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema
        self.epochs = epochs
        self.device = device
        self.img_size = img_size
        self.low_size = low_size
        scale_factor = img_size // low_size
        self.netG = init_net(PonoSPadaInGenerator(dim_in, img_size=img_size, low_size=low_size), 'kaiming')
        self.netD = init_net(Discriminator(img_size=img_size, low_size=low_size), 'kaiming')

        if use_ema :
            import copy
            self.AverageG = copy.deepcopy(self.netG)
            self.update_average(self.AverageG, self.netG, beta=0.)

        self.optG = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0., 0.99))
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=0.004, betas=(0., 0.99))
        #wandb.init(project='Paper2', name='PonoSPadaIn-cats-1')

        self.advLoss = R1Loss_STE(self.netD)
        self.ReconLoss = nn.L1Loss()

    def train_step(self, x_a, x_b):
        
        self.optD.zero_grad()
        
        x_a_low = downsample(x_a, self.low_size)
        x_b_low = downsample(x_b, self.low_size)
        
        x_ab = self.netG(x_a, x_b_low)
        x_a_recon = self.netG(x_ab, x_a_low)

        x_ba = self.netG(x_b, x_a_low)
        x_b_recon = self.netG(x_ba, x_b_low)

        self.d_loss = self.advLoss.dis_loss(x_b, x_b_low, x_ab.detach()) \
                    + self.advLoss.dis_loss(x_a, x_a_low, x_ba.detach()) \
                    + self.advLoss.dis_loss(x_a, x_a_low, x_a_recon.detach()) \
                    + self.advLoss.dis_loss(x_b, x_b_low, x_b_recon.detach())


        self.d_loss.backward()
        self.optD.step()

        self.optG.zero_grad()
        self.recon_loss = self.ReconLoss(x_a, x_a_recon) + self.ReconLoss(x_b, x_b_recon)
        self.g_adv_loss = self.advLoss.gen_loss(x_ab, x_b_low) \
                        + self.advLoss.gen_loss(x_ba, x_a_low) \
                        + self.advLoss.gen_loss(x_a_recon, x_a_low) \
                        + self.advLoss.gen_loss(x_b_recon, x_b_low) 

        self.g_loss = self.g_adv_loss + self.recon_loss

        self.g_loss.backward()
        self.optG.step()

        self.fake_ba = x_ba
        self.fake_ab = x_ab
        self.recon_a = x_a_recon
        self.recon_b = x_b_recon
        self.x_a = x_a
        self.x_b = x_b
        
        if self.use_ema :
            self.update_average(self.AverageG, self.netG, 0.999)

    def train(self, dataloader):
        self.step = 0

        self.save_model()
        for _ in range(self.epochs):
            for (x, _) in dataloader:
                
                x = x
                x_a, x_b = torch.split(x, x.size(0)//2, dim=0)
                x_a = x_a.to(self.device)
                x_b = x_b.to(self.device)

                self.step += 1 
                self.train_step(x_a, x_b)


                if self.step % 100 == 0 :
                    WandbLogger.add_scalar('d_loss', self.d_loss.item(), self.step)
                    WandbLogger.add_scalar('g_loss', self.g_loss.item(), self.step)
                    WandbLogger.add_scalar('g adv loss', self.g_adv_loss.item(), self.step)
                    WandbLogger.add_scalar('g recon loss', self.recon_loss.item(), self.step)

                if self.step % 1000 == 0 :
                    WandbLogger.add_image('real a', self.x_a, self.step, nb_imgs=4)
                    WandbLogger.add_image('real b', self.x_b, self.step, nb_imgs=4)
                    WandbLogger.add_image('recon a', self.recon_a, self.step, nb_imgs=4)
                    WandbLogger.add_image('recon b', self.recon_b, self.step, nb_imgs=4)
                    WandbLogger.add_image('fake a->b', self.fake_ab, self.step, nb_imgs=4)
                    WandbLogger.add_image('fake b->a', self.fake_ba, self.step, nb_imgs=4)

                if self.step % 10000 == 0:
                    self.save_model()


    def save_model(self):
        dict_save = {
            'Gen' : self.netG.state_dict(), 
            'Dis' : self.netD.state_dict(),               
            'optG' : self.optG.state_dict(),
            'optD' : self.optD.state_dict()}
        if self.use_ema : 
            dict_save['AverageG'] = self.AverageG.state_dict()
        torch.save(dict_save, self.checkpoint_path + '.pt')
        print(f'MODEL Saved at step {self.step}')