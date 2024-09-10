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
from torch.utils.data import DataLoader
import argparse
# import pretrainedmodels
from Attacker import Attacker
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
from torch.utils import data
import pandas as pd
# num_workers=0
seed = 42
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images', help='Input directory with images.')
parser.add_argument('--model_dir', type=str, default='./models/', help='model directory.') 
parser.add_argument('--model_name', type=str, default='tf2torch_inception_v3', help='source model name.') # 
parser.add_argument('--output_dir', type=str, default='./outputs/MI_ssa_jpeg/', help='Output directory with adversarial images.') #
parser.add_argument("--batch_size", type=int, default=50, help="How many images process at one time.") #
parser.add_argument("--N", type=int, default=5, help="The copy number ") # 
parser.add_argument('--bound', type=float, default= 45 , help='random noise bound')
parser.add_argument('--line', type=int, default= 280 , help='length parameter')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.") # 
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

parser.add_argument("--beta", type=float, default=1.5, help="beta")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
        TfNormalize('torch'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model

# def grad_X(x, y, model):
#     model.eval()
#     logits = model(x)
#     cross_entropy = F.cross_entropy(logits, y)
#     grad = torch.autograd.grad(cross_entropy, x)[0]
#     return grad

def grad_X(image, label, model):
    # 创建一个新的张量，并设置 requires_grad 属性为 True
    image_var = V(image, requires_grad=True)
    output = model(image_var)
    loss = F.cross_entropy(output[0], label)
    model.zero_grad()
    loss.backward()
    gradient =torch.autograd.grad(loss, image_var)[0]
    return gradient

class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        # ImageID = img_obj['ImageId'] + '.JPEG'
        Truelabel = img_obj['TrueLabel'] 
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)

class MIFGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(MIFGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        alpha = self.config['eps'] / self.config['attack_steps'] 
        decay = 1.0
        x_adv = x.detach().clone()
        momentum = torch.zeros_like(x_adv, device=x.device)
        if self.config['random_init'] :
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
            x_adv = torch.clamp(x_adv,*self.clamp)


        for step in range(self.config['attack_steps']):
            x.requires_grad=True
            logit = self.model(x)
            logit=logit[0]
            if self.target is None:
                cost = -F.cross_entropy(logit, y)
            else:
                cost = F.cross_entropy(logit, target)
            grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum*decay
            momentum = grad
            x_adv = x - alpha*grad.sign()
            a = torch.clamp(x - self.config['eps'], min=0)
            b = (x_adv >= a).float()*x_adv + (a > x_adv).float()*a
            c = (b > x + self.config['eps']).float() * (x + self.config['eps']) + (
                x + self.config['eps'] >= b
            ).float() * b
            x = torch.clamp(c, max=1).detach()
        x_adv = torch.clamp(x, *self.clamp)
        return x_adv

def main():

    ## model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    ##                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    # model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    #                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().to(device))

    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)  #  
    attack_config = {
            'attack': 0,
            'eps' : 8/255.0,
            'attack_steps': 10,
            'attack_lr':1 / 255.0,
            'random_init': False,
        }
    MI=MIFGSM(model,  attack_config )
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.to(device)
        images = images.to(device)              
        # images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        # images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = MI(images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()
