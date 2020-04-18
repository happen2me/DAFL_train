import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=10, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='cache/models/')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


def run():
    generator = Generator().to(device)

    teacher = torch.load(opt.teacher_dir + 'teacher').to(device)
    teacher.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    teacher = nn.DataParallel(teacher)
    generator = nn.DataParallel(generator)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)

    # ----------
    #  Training
    # ----------
    for epoch in range(opt.n_epochs):

        for i in range(120):
            generator.train()
            z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).to(device)
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T,pred)
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
            loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
            loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
            loss.backward()
            optimizer_G.step()
            if i == 1:
                print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item()))

        torch.save(generator.state_dict(), opt.output_dir + "generator_only.pt")
        print("generator saved at ", opt.output_dir + "generator_only.pt")


if __name__ == "__main__":
    run()