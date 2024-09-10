from typing import Callable
from attack2 import Attack
import torch
import torch.nn as nn
from attack import Attack
import os
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
parser.add_argument('--output_dir', type=str, default='./outputs/VMI/', help='Output directory with adversarial images.') #
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

class Admix(Attack):
    """The Admix attack.

    From the paper 'Admix: Enhancing the Transferability of Adversarial Attacks',
    https://arxiv.org/abs/2102.00436

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        portion: Portion for the mixed image. Defaults to 0.2.
        size: Number of randomly sampled images. Defaults to 3.
        num_classes: Number of classes of the dataset used. Defaults to 1001.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        portion: float = 0.2,
        size: int = 3,
        num_classes: int = 1001,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(normalize, device)

        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.portion = portion
        self.size = size
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform Admix on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        g = torch.zeros_like(x)
        x_adv = x.clone().detach()

        scales = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        # Admix + MI-FGSM
        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Add delta to original image then admix
            x_admix = self.admix(x_adv)
            x_admixs = torch.cat([x_admix * scale for scale in scales])

            # Compute loss
            outs = self.model(self.normalize(x_admixs))

            # One-hot encode labels for all admixed images
            one_hot = nn.functional.one_hot(y, self.num_classes)
            one_hot = torch.cat([one_hot] * 5 * self.size).float()

            loss = self.lossfn(outs[0], one_hot)

            if self.targeted:
                loss = -loss

            # Gradients
            grad = torch.autograd.grad(loss, x_admixs)[0]

            # Split gradients and compute mean
            grads = torch.tensor_split(grad, 5, dim=0)
            grads = [g * s for g, s in zip(grads, scales, strict=True)]
            grad = torch.mean(torch.stack(grads), dim=0)

            # Gather gradients
            grads = torch.tensor_split(grad, self.size)
            grad = torch.sum(torch.stack(grads), dim=0)

            # Apply momentum term
            g = self.decay * g + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )

            # Update perturbed image
            x_adv = x_adv.detach() + self.alpha * g.sign()
            x_adv = x + torch.clamp(x_adv - x, -self.eps, self.eps)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return x_adv

    def admix(self, x: torch.Tensor) -> torch.Tensor:
        def x_admix(x: torch.Tensor) -> torch.Tensor:
            return x + self.portion * x[torch.randperm(x.shape[0])]

        return torch.cat([(x_admix(x)) for _ in range(self.size)])

def main():

    model = get_model(opt.model_name, opt.model_dir) #  [-1,1]
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)  #  
    VNI=Admix(model=model,device=device,normalize=transforms)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.to(device)
        images = images.to(device)              
        adv_img = VNI(images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()