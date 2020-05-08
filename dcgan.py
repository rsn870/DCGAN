

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import  tqdm 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as utils
import visdom
import os 
from visualiser_utils import *


#########WINDOWS##############
"""
vis = visdom.Visdom(port=8097,env='main')
vis.close(env='main')

D_loss_window = make_single_line_plot(window=None,visdom_object =vis,Y=None,X=None,env='main',
                          first_time=True,loss_name='Discriminator_Loss')

G_loss_window = make_single_line_plot(window=None,visdom_object =vis,Y=None,X=None,env='main',
                          first_time=True,loss_name='Generator_Loss')

Image_real = make_images_plot(visdom_object=vis,env='main',first_time=True,nrow=3)

Image_fake = make_images_plot(visdom_object=vis,env='main',first_time=True,nrow=3)
"""
######################################################






transform = transforms.Compose(
    [transforms.Resize((64,64)) , transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

checkpoint_parent_dir = './SAVED_MODELS'
dataiter = iter(trainloader)
images, labels = dataiter.next()


if torch.cuda.is_available() :
  device = torch.device("cuda:0")
else :
  device = "cpu"

nz = 100
nc = 3
ngf = 2
ndf = 2

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self , x) :

      output = self.main(x)

      return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

      output = self.main(x)
      return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = Generator()
netG = netG.to(device)
netG.apply(weights_init)

netD = Discriminator()
netD = netD.to(device)
netD.apply(weights_init)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

G_solver = optim.Adam(netG.parameters(), lr=0.01)
D_solver = optim.Adam(netD.parameters(), lr=0.01)

for epoch in tqdm(range(100000)):
   for i, data in enumerate(trainloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        D_real = netD(real_cpu)

        D_x = D_real.mean().item()

        noise = torch.randn(64, nz, 1, 1, device=device)
        fake = netG(noise)

        D_fake = netD(fake)

        D_G_z1 = D_fake.mean().item()

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

        D_loss.backward(retain_graph= True) 
        D_solver.step() 

        
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        netD.zero_grad()
        netG.zero_grad()

        output = netD(fake)
        D_G_z2 = output.mean().item()
        G_loss = -torch.mean(output)
        G_loss.backward(retain_graph= True)
        G_solver.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, 100000, i, len(trainloader),
                 D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 10 == 0:
            print('saving the output')
            utils.save_image(real_cpu,'output/real/real_samples.png',normalize=True)
            fake = netG(fixed_noise)

            directory = str(epoch)

            #parent_dir = "output"

            #path = os.path.join(parent_dir , directory)

            #os.mkdir(path)
            
            
            
            utils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch),normalize=True)

            """
            
            Y = D_loss.view(-1,1)

            X = torch.Tensor([i]).view(-1,1)


            make_single_line_plot(window=D_loss_window,
                                visdom_object=vis,Y=Y,X=X,env='main',first_time=False)

            Y = G_loss.view(-1,1)


            make_single_line_plot(window=G_loss_window,
                                visdom_object=vis,Y=Y,X=X,env='main',first_time=False)

            fake = fake.detach().cpu()



            make_images_plot(image_tensor=fake,visdom_object=vis,image_window=Image_fake,first_time=False,nrow=3)
            make_images_plot(image_tensor=real_cpu,visdom_object=vis,image_window=Image_real,first_time=False,nrow=3)
            
            """
            
           



        
        
        
        if i % 100 == 0 :
            
            
            
        
        
        
        
        
        
        
        
        torch.save(netG.state_dict(), 'weights/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))
        
        #Uncomment the visdom if plots are required for some sample losses  









