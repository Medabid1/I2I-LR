import torch 
import wandb

from torchvision import utils


class WandbLogger:

    @staticmethod
    def add_image(tag, image, step, nb_imgs=None):
        if nb_imgs != None :
            image = image[:nb_imgs]
        grid_im = utils.make_grid(image, normalize=True)
        d = {tag: wandb.Image(grid_im.cpu().data)}
        wandb.log(d, step=step)
    
    @staticmethod
    def add_scalar(tag, scalar, step):
        d = {tag: scalar}
        wandb.log(d, step=step)

    @staticmethod
    def add_list_images(tag, images, step):
        for i in range(len(images)):
            grid_im = utils.make_grid(images[i], normalize=True)
            d = {tag + str(i): wandb.Image(grid_im.cpu().data)}
            wandb.log(d , step=step)