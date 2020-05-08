import visdom
import torch
import numpy as np
import pandas as pd
# from .types_ import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

def make_line_plot(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        x=torch.Tensor(list(range(0,10))).view(-1,1)
        
        plt2=visdom_object.line(Y=torch.randn(10,2).zero_(),X=torch.cat([x,x],dim=1).zero_(),env=env,opts=dict(
                legend=['G_loss','D_loss'],
                showlegend=True,
                title='Loss',
                xlabel='Iterations',
                ylabel='Loss'))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)



def make_line_plot_L1(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='Loss',title='L1_loss',legend=['Loss']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)



def make_line_plot_Wasserstein(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='Distance',title='Wasserstein_distance',legend=['distance']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)




def make_line_plot_G_loss(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='Loss',title='G_loss',legend=['Loss']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)


def make_line_plot_GP(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='GP',title='Gradient_penalty',legend=['GP']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)



def make_images_plot(image_window=None,visdom_object =None,image_tensor=None,env='main',first_time=False,nrow=5):
    if (first_time):
        window = visdom_object.images(torch.rand((6,1,128,128)),nrow=nrow,env=env)
        
        return window
    else:
        visdom_object.images(image_tensor,nrow=nrow,env='main',win=image_window)


def make_line_plot_perceptual(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='Loss',title='Perceptual_loss',legend=['Loss']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)



def plot_ssim(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='ssim',title='SSIM_constant',legend=['ssim']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)

def plot_psnr(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='psnr',title='PSNR',legend=['psnr']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)


def plot_mssim(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel='mssim',title='MSSIM',legend=['MSSIM']))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)


def make_single_line_plot(window=None,visdom_object =None,Y=None,X=None,env='main',first_time=False,loss_name='L1_loss'):
    if (first_time):
        
        
        plt2=visdom_object.line(Y=torch.zeros((1)).cpu(),X=torch.zeros((1)).cpu(),
            opts=dict(xlabel='Iterations',ylabel=loss_name,title=loss_name,legend=[loss_name]))
        
        return plt2
    else:
        visdom_object.line(Y=Y,X=X,update='append',win=window,env = env)
