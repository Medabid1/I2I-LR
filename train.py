import os 
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np 
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


import wandb

os.environ['WANDB_API_KEY'] = '' 

from models.PonoSPAdaInModel import Model

import argparse 





def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataPath", action="store", type=str,
                        default=None,
                        help="path")
    
    parser.add_argument("--checkpoint", action="store", type=str,
                        default=None,
                        help="checkpoint_path")

    parser.add_argument("--use_ema", action="store", type=bool,
                        default=True,
                        help="ema")
    parser.add_argument("--img_size", action="store", type=int,
                        default=128,
                        help="img_size")
    parser.add_argument("--low_size", action="store", type=int,
                        default=8,
                        help="img_size")
    parser.add_argument("--dim_in", action="store", type=int,
                        default=512,
                        help="dim_in")
    parser.add_argument("--epochs", action="store", type=int,
                        default=100,
                        help="epochs")
    parser.add_argument("--batch_size", action="store", type=int,
                        default=8,
                        help="batchSize")

    parser.add_argument("--device", action="store", type=str,
                        default='cuda',
                        help="device")
    
    args = parser.parse_args()
    return args


args = parse_arguments()


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path = ''
data = ImageFolder(root= path, transform= transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))


dataloader = DataLoader(data, args.batch_size, shuffle=True, drop_last=True)

model = Model(args.img_size, args.low_size, args.dim_in, epochs=args.epochs, checkpoint_path=args.checkpoint, device=args.device, use_ema=args.use_ema)
model.train(dataloader)
