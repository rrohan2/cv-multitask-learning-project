# import sys
# import os
# sys.path.append(os.path.abspath('.'))
# import torch
# import numpy as np
# import cv2
# import os
# from PIL import Image
# import glob
# import matplotlib.pyplot as plt
# from multitask_project.multitask_model import HydraNet
# from multitask_project.utils import prepare_img, depth_to_rgb
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load model
# model = HydraNet(num_tasks=2, num_classes=6).to(device)
# model.eval()
# ckpt = torch.load('checkpoints/ExpKITTI_joint.ckpt')
# model.load_state_dict(ckpt['state_dict'])

# # Load CMAP
# CMAP = np.load('data/cmap_kitti.npy')

# def apply_cmap(segm):
#     return (CMAP[segm] * 255).astype(np.uint8)

# def pipeline(img, model):
#     with torch.no_grad():
#         img_input = prepare_img(img).transpose(2, 0, 1)[None].astype(np.float32)
#         img_tensor = torch.from_numpy(img_input).to(device)
#         segm, depth = model(img_tensor)

#         segm = segm.argmax(dim=1).squeeze().cpu().numpy()
#         segm = apply_cmap(segm)

#         depth = torch.abs(depth.squeeze()).cpu().numpy()
#         return depth, segm

# # Run inference
# video_files = sorted(glob.glob("data/*.png"))
# result_video = []

# for img_path in video_files:
#     img = np.array(Image.open(img_path))
#     h, w, _ = img.shape
#     depth, segm = pipeline(img, model)

#     combined = cv2.vconcat([img, segm, depth_to_rgb(depth)])
#     result_video.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

# # Save the video
# os.makedirs('outputs/videos', exist_ok=True)
# out = cv2.VideoWriter('outputs/videos/out.mp4',
#                       cv2.VideoWriter_fourcc(*'MP4V'), 15, (w, 3 * h))
# for frame in result_video:
#     out.write(frame)
# out.release()
# print("Video saved at outputs/videos/out.mp4")

import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import numpy as np
import cv2
import glob
from PIL import Image
from multitask_project.multitask_model import HydraNet
from multitask_project.utils import prepare_img, depth_to_rgb
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = HydraNet(num_tasks=2, num_classes=6).to(device)
model.eval()

ckpt = torch.load('checkpoints/ExpKITTI_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

# Load segmentation color map
CMAP = np.load('data/cmap_kitti.npy')

def apply_cmap(segm):
    return (CMAP[segm] * 255).astype(np.uint8)

# def pipeline(img, model):
#     with torch.no_grad():
#         img_input = prepare_img(img).transpose(2, 0, 1)[None].astype(np.float32)
#         img_tensor = torch.from_numpy(img_input).to(device)
#         segm, depth = model(img_tensor)

#         segm = segm.argmax(dim=1).squeeze().cpu().numpy()
#         segm = apply_cmap(segm)

#         depth = torch.abs(depth.squeeze()).cpu().numpy()
#         return depth, segm

def pipeline(img):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        segm = cv2.resize(segm[0, :6].cpu().data.numpy().transpose(1, 2, 0),
                        img.shape[:2][::-1],
                        interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                        img.shape[:2][::-1],
                        interpolation=cv2.INTER_CUBIC)
        segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
        depth = np.abs(depth)
        return depth, segm
     

# Run pipeline on each image
video_files = sorted(glob.glob("data/*.png"))
# result_video = []

# for img_path in video_files:
#     img = np.array(Image.open(img_path))
#     h, w, _ = img.shape

#     depth, segm = pipeline(img, model)

#     # Resize + stack safely
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     segm = cv2.resize(segm, (w, h)).astype(np.uint8)
#     depth_colored = cv2.resize(depth_to_rgb(depth), (w, h)).astype(np.uint8)

#     combined = cv2.vconcat([img, segm, depth_colored])
#     result_video.append(combined)

result_video = []
for idx, img_path in enumerate(video_files):
    image = np.array(Image.open(img_path))
    h, w, _ = image.shape 
    depth, seg = pipeline(image)
    result_video.append(cv2.cvtColor(cv2.vconcat([image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))

# Write output video
os.makedirs('outputs/videos', exist_ok=True)
out = cv2.VideoWriter('outputs/videos/out.mp4',
                      cv2.VideoWriter_fourcc(*'MP4V'), 15, (w, 3 * h))

for frame in result_video:
    out.write(frame)
out.release()
print("Video saved at outputs/videos/out.mp4")
