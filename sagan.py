

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
from spectral import SpectralNorm

#########WINDOWS##############

vis = visdom.Visdom(port=8097,env='main')
vis.close(env='main')

D_loss_window = make_single_line_plot(window=None,visdom_object =vis,Y=None,X=None,env='main',
                          first_time=True,loss_name='Discriminator_Loss')

G_loss_window = make_single_line_plot(window=None,visdom_object =vis,Y=None,X=None,env='main',
                          first_time=True,loss_name='Generator_Loss')

Image_real = make_images_plot(visdom_object=vis,env='main',first_time=True,nrow=3)

Image_fake = make_images_plot(visdom_object=vis,env='main',first_time=True,nrow=3)

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



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size = 4, image_size=64, z_dim=10, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=4, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2




#def weights_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1:
       # m.weight.data.normal_(0.0, 0.02)
    #elif classname.find('BatchNorm') != -1:
       #m.weight.data.normal_(1.0, 0.02)
       #m.bias.data.fill_(0)
nz = 10
netG = Generator()
netG = netG.to(device)
#netG.apply(weights_init)

netD = Discriminator()
netD = netD.to(device)
#netD.apply(weights_init)

fixed_noise = torch.randn(4, nz, 1, 1, device=device)

G_solver = optim.Adam(netG.parameters(), lr=0.01)
D_solver = optim.Adam(netD.parameters(), lr=0.01)

for epoch in tqdm(range(300)):
   for i, data in enumerate(trainloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        D_real , p1 , p2 = netD(real_cpu)

        D_x = D_real.mean().item()

        noise = torch.randn(4, nz, 1, 1, device=device)
        fake ,p1 , p2= netG(noise)

        D_fake , p1 , p2 = netD(fake)

        D_G_z1 = D_fake.mean().item()

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
	if i > 0 :
	    D_loss.backward(retain_graph= True)
	else :
	    D_loss.backward(retain_graph= True)
        D_solver.step()      
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
	
        netD.zero_grad()
        netG.zero_grad()

        output , b , g = netD(fake)
        D_G_z2 = output.mean().item()
        G_loss = -torch.mean(output)
        if i > 0 :
            G_loss.backward(retain_graph= True)
        else :
            G_loss.backward(retain_graph= True)
 
        G_solver.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, 100000, i, len(trainloader),
                 D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))
        
        if i % 10 == 0:
            print('saving the output')
            
            utils.save_image(real_cpu,'output/real/real_samples.png',normalize=True)
            

            fake , a , m = netG(fixed_noise)


          
            
            utils.save_image(fake.detach(),'output/fake_samples_epoch_%03d.png' % (epoch),normalize=True)

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

            
            
            torch.save(netG.state_dict(), 'sa_weights/netG_epoch_%d.pth' % (epoch))
            torch.save(netD.state_dict(), 'sa_weights/netD_epoch_%d.pth' % (epoch))













