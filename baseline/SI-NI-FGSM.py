import torch
import torch.nn as nn
from attack import Attack
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
from dct import *
from Normalize import Normalize, TfNormalize
from torch import nn
from torch.utils import data
import pandas as pd
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
seed = 42
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='/MinierData/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images', help='Input directory with images.')
parser.add_argument('--model_dir', type=str, default='/MinierData/tangjiawei/FSD-MIM-and-NPGA/models', help='model directory.') 
parser.add_argument('--model_name', type=str, default='tf2torch_inception_v3', help='source model name.') # 
parser.add_argument('--output_dir', type=str, default='./outputs/SI-NI-FGSM/', help='Output directory with adversarial images.') #
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        TfNormalize('tensorflow'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model


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

class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, m=5):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay * self.alpha * momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                outputs = self.get_logits(nes_images)
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs[0], target_labels)
                else:
                    cost = loss(outputs[0], labels)
                adv_grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
def main():

    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)  #  
    sini=SINIFGSM(model=model)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.to(device)
        images = images.to(device)              
        adv_img = sini(images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()
