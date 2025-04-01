import sys
import os
sys.path.append(os.path.abspath("."))
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import cv2
from multitask_project.multitask_model import HydraNet
from multitask_project.utils import prepare_img


import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import define_mobilenet
from .decoder import define_lightweight_refinenet, _make_crp

 
class HydraNet(nn.Module):
    def __init__(self, num_tasks=2, num_classes=6):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes

        # Bind _make_crp so it can be used in define_lightweight_refinenet
        self._make_crp = _make_crp.__get__(self)

        # Build encoder and decoder
        define_mobilenet(self)
        define_lightweight_refinenet(self)

    def forward(self, x):
        # --- Encoder ---
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)

        # --- Decoder ---
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=False)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=False)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=False)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # --- Heads ---
        out_segm = self.relu(self.pre_segm(l3))
        out_segm = self.segm(out_segm)

        out_depth = self.relu(self.pre_depth(l3))
        out_depth = self.depth(out_depth)

        if self.num_tasks == 3:
            out_normal = self.relu(self.pre_normal(l3))
            out_normal = self.normal(out_normal)
            return out_segm, out_depth, out_normal
        else:
            return out_segm, out_depth


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIR = 'project_midpoint/seg_depth/data/seg'
OUTPUT_DIR = 'project_midpoint/seg_depth/predictions/seg'
CMAP_PATH = 'data/cmap_kitti.npy'
CKPT_PATH = 'checkpoints/ExpKITTI_joint.ckpt'

os.makedirs(OUTPUT_DIR, exist_ok=True)


model = HydraNet(num_tasks=2, num_classes=6).to(device)
model.eval()
ckpt = torch.load(CKPT_PATH)
model.load_state_dict(ckpt['state_dict'])


CMAP = np.load(CMAP_PATH)


def pipeline(img):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        img_var = img_var.to(device)

        segm, _ = model(img_var)
        segm = segm[0].cpu().data.numpy()
        segm = cv2.resize(segm.transpose(1, 2, 0), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

        segm_raw = segm.argmax(axis=2).astype(np.uint8)
        return segm_raw


image_paths = sorted(glob.glob(f'{INPUT_DIR}/*.png'))

for img_path in image_paths:
    img = np.array(Image.open(img_path))
    filename = os.path.splitext(os.path.basename(img_path))[0]

    seg_mask = pipeline(img)

    seg_color = (CMAP[seg_mask] * 255).astype(np.uint8)  
    cv2.imwrite(f'{OUTPUT_DIR}/{filename}.png', cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))


print("PNG masks saved to:", OUTPUT_DIR)
