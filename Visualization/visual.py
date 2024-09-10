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
from loader import ImageNet,ImageNet_ssa
from torch.utils.data import DataLoader
import argparse
# import pretrainedmodels
import matplotlib.pyplot as plt
import seaborn as sns
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
seed = 42
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
model=get_model('tf2torch_adv_inception_v3','/home/liushifa/tangjiawei/FSD-MIM-and-NPGA/models')

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

from ..loader import ImageNet_SFA

#生成的对抗图像
X = ImageNet_SFA('/home/liushifa/tangjiawei/paper/outputs/tf2torch_adv_inception_v3_to_norM_MI_two', '/home/liushifa/tangjiawei/SSA-c955cf07c8372bfc4e9f17e647042e027f9f3b1d/dataset/images.csv', transforms)
data_loader = DataLoader(X, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)  #  

grad_all = 0
for images, images_ID,  gt_cpu in tqdm(data_loader):    
    gt = gt_cpu.to(device)
    images = images.to(device)
    img_dct = dct_2d(images)
    img_dct = V(img_dct, requires_grad = True)
    img_idct = idct_2d(img_dct)

    output_ = model(img_idct)
    loss = F.cross_entropy(output_[0], gt)
    loss.backward()
    grad = img_dct.grad.data
    grad = grad.mean(dim = 1).abs().sum(dim = 0).cpu().numpy()
    grad_all = grad_all + grad
    
x = grad_all / 1000.0
x = (x - x.min()) / (x.max() - x.min())
g1 = sns.heatmap(x, cmap="rainbow")
g1.set(yticklabels=[])  # remove the tick labels
g1.set(ylabel=None)  # remove the axis label
g1.set(xticklabels=[])  # remove the tick labels
g1.set(xlabel=None)  # remove the axis label
g1.tick_params(left=False)
g1.tick_params(bottom=False)
sns.despine(left=True, bottom=True)
plt.gcf().set_dpi(300)  
plt.show()
plt.savefig("fig.png")