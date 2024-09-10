"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI,gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
from loader import ImageNet,ImageNet_SFA
from torch.utils.data import DataLoader
import argparse

from dct import *
from Normalize import Normalize, TfNormalize
from torch import nn
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
# num_workers=0
seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--model_dir', type=str, default='./models', help='model directory.') 
parser.add_argument('--model_name', type=str, default='tf2torch_inception_v3', help='source model name.') # 
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.') #
parser.add_argument("--batch_size", type=int, default=30, help="How many images process at one time.") #
parser.add_argument("--N", type=int, default=5, help="The copy number ") # 
parser.add_argument('--bound', type=float, default= 45 , help='random noise bound')
parser.add_argument('--line', type=int, default= 280 , help='length parameter')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=5, help="Number of iterations.") # 
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

parser.add_argument("--beta", type=float, default=3.5, help="beta")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1: # 
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square) # 
    return nor_grad

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))  
        img.save(output_dir + name)
        
def save_image1(images, names, output_dir):  
    """Save the images as PNG files with modified names."""  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
      
    for i, name in enumerate(names):  
        # 将name的后缀改为.png  
        png_name = os.path.splitext(name)[0] + '.JPEG'  
        img = Image.fromarray(images[i].astype('uint8'))  
        img.save(os.path.join(output_dir, png_name))

T_kernel = gkern(7, 3) # 3,1,7,7

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model


def grad_X(image, label, model):
    # 创建一个新的张量，并设置 requires_grad 属性为 True
    image_var = V(image, requires_grad=True)
    output = model(image_var)
    loss = F.cross_entropy(output[0], label)
    model.zero_grad()
    loss.backward()
    gradient = image_var.grad.data
    return gradient


def SFA(images, gt, model, min, max):
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = opt.num_iter_set
    eps =  opt.max_epsilon / 255.0
    alpha_hyprt = eps * opt.beta
    alpha = eps / num_iter
    x = images.clone() # clone
    x_adv = x.clone()
    noise = torch.zeros_like(x)
    g_i = []
    x_advs = []
    for iter in range(0, num_iter):     
        x_advs.append(x_adv) 
        if(iter == 0):
            x_neighbor_0 = x
        else:    
            x_neighbor_0 = x_advs[iter - 1] + alpha * g_i[iter - 1]
        x_last = x_neighbor_0.clone()
        x_last.to(device)
        gs = torch.zeros_like(x)    
        gs = gs + grad_X(x_neighbor_0, gt, model)
        for j in range(0, 20):    # 
            x_neighbor = x_last + torch.FloatTensor(x.size()).uniform_(-alpha_hyprt, alpha_hyprt).to(device)
            x_neighbor.to(device)
            gs = gs + grad_X(x_neighbor, gt, model)
        g_ = (gs + grad_X(x_adv, gt, model)) / (20.0 + 2) #在iter上进行比较
        g_ = g_ / torch.abs(g_).mean([1, 2, 3], keepdim=True)
        g_i.append(g_)    
        noise = momentum * noise + g_
        x_adv = x_adv +  alpha * torch.sign(noise)
        x_adv = clip_by_tensor(x_adv, min, max)
    return x_adv.detach()
##################################################################


def main():
  
    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]
    X = ImageNet_SFA(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)  #  
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.to(device)
        images = images.to(device)              
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = SFA(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image1(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()
