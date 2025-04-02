import sys
import os
sys.path.append(os.path.abspath("."))
import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image
from torch.autograd import Variable
from multitask_project.multitask_model import HydraNet
from multitask_project.utils import prepare_img

# --- Paths ---
SEG_INPUT_DIR = 'project_midpoint/seg_depth/data/seg'
DEPTH_INPUT_DIR = 'project_midpoint/seg_depth/data/depth'
SEG_OUTPUT_DIR = 'project_midpoint/seg_depth/predictions/seg'
DEPTH_OUTPUT_DIR = 'project_midpoint/seg_depth/predictions/depth'
CMAP_PATH = 'data/cmap_kitti.npy'
CKPT_PATH = 'checkpoints/ExpKITTI_joint.ckpt'

os.makedirs(SEG_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEPTH_OUTPUT_DIR, exist_ok=True)

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HydraNet(num_tasks=2, num_classes=6).to(device)
model.eval()
ckpt = torch.load(CKPT_PATH)
model.load_state_dict(ckpt['state_dict'])

# --- Load colormap ---
CMAP = np.load(CMAP_PATH)

# --- Segmentation pipeline ---
def segmentation_pipeline(img):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None])).float().to(device)
        segm, _ = model(img_var)
        segm = segm[0].cpu().data.numpy()
        segm = cv2.resize(segm.transpose(1, 2, 0), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        segm_raw = segm.argmax(axis=2).astype(np.uint8)
        return segm_raw

# --- Depth pipeline ---
def depth_pipeline(img):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None])).float().to(device)
        _, depth = model(img_var)
        depth = depth[0, 0].cpu().data.numpy()
        depth = cv2.resize(depth, img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        return np.abs(depth)

# --- Run segmentation ---
seg_image_paths = sorted(glob.glob(f'{SEG_INPUT_DIR}/*.png'))
for img_path in seg_image_paths:
    img = np.array(Image.open(img_path))
    filename = os.path.splitext(os.path.basename(img_path))[0]
    seg_mask = segmentation_pipeline(img)
    seg_color = (CMAP[seg_mask] * 255).astype(np.uint8)
    cv2.imwrite(f'{SEG_OUTPUT_DIR}/{filename}.png', cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))

# --- Run depth ---
depth_image_paths = sorted(glob.glob(f'{DEPTH_INPUT_DIR}/*.png'))
for img_path in depth_image_paths:
    img = np.array(Image.open(img_path))
    filename = os.path.splitext(os.path.basename(img_path))[0]
    depth_map = depth_pipeline(img)
    depth_scaled = (depth_map * 256.0).astype(np.uint16)
    cv2.imwrite(f'{DEPTH_OUTPUT_DIR}/{filename}.png', depth_scaled)

print("Segmentation saved to:", SEG_OUTPUT_DIR)
print("Depth saved to:", DEPTH_OUTPUT_DIR)
