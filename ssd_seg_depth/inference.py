# import os
# import sys
# sys.path.append(os.path.abspath("."))
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from ssd_model import HydraNet, SSD300MB9, depth_to_rgb, drawboxes
# import torch.nn.functional as F

# # CONFIG
# IMG_DIR = "data"
# VIDEO_PATH = "output/ssd_output.mp4"
# CKPT_PATH = "weights/ExpKITTI_joint.ckpt"
# SSD_PATH = "weights/SSDMB9_001.tar"
# CMAP_PATH = "cmap_kitti.npy"
# NUM_CLASSES = 6

# # HELPER
# IMG_SCALE  = 1./255
# IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
# IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
# def prepare_img(img):
#     return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

# # SETUP
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CMAP = np.load(CMAP_PATH)

# # Build and load model
# hydranet = HydraNet()
# hydranet.define_mobilenet()
# hydranet.define_lightweight_refinenet()
# hydranet.object_detection_head = 1
# # hydranet.load_state_dict(torch.load(CKPT_PATH, map_location=device))
# ckpt = torch.load(CKPT_PATH, map_location=device)
# hydranet.load_state_dict(ckpt['state_dict'])

# hydranet.eval().to(device)

# ssd_model = SSD300MB9(num_classes=9, base_model=hydranet)
# # ssd_model.load_state_dict(torch.load(SSD_PATH, map_location=device))
# ckpt = torch.load(SSD_PATH, map_location=device)
# if "state_dict" in ckpt:
#     ssd_model.load_state_dict(ckpt["state_dict"], strict=False)
# else:
#     ssd_model.load_state_dict(ckpt, strict=False)

# ssd_model.eval().to(device)

# # Detection function
# def detect(model, img_pil, device, min_score, max_overlap, top_k):
#     from torchvision import transforms
#     resize = transforms.Resize((300, 300))
#     to_tensor = transforms.ToTensor()
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     image = normalize(to_tensor(resize(img_pil))).unsqueeze(0).to(device)
#     with torch.no_grad():
#         segm, depth, locs_pred, cls_pred = model(image)
#         boxes, labels, scores = model.detect(locs_pred, cls_pred, min_score, max_overlap, top_k)
#     return segm, depth, img_pil, (boxes[0], labels[0], scores[0])

# # Inference
# image_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".png")])
# frames = []

# for path in tqdm(image_paths, desc="Running SSD Inference"):
#     img = Image.open(path).convert("RGB")
#     w, h = img.size
#     segm, depth, annotated, (boxes, labels, scores) = detect(ssd_model, img, device, min_score=0.4, max_overlap=0.4, top_k=200)

#     segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_CUBIC)
#     segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)

#     depth = cv2.resize(depth[0, 0].cpu().data.numpy(), (w, h), interpolation=cv2.INTER_CUBIC)
#     depth_rgb = depth_to_rgb(depth)

#     annotated = np.array(drawboxes(np.array(annotated), boxes, labels, scores))
#     segm = drawboxes(segm, boxes, labels, scores)
#     depth_rgb = drawboxes(depth_rgb, boxes, labels, scores)

#     stacked = np.vstack([annotated, segm, depth_rgb])
#     frames.append(stacked)

# # Save Video
# os.makedirs("output", exist_ok=True)
# w, h = frames[0].shape[1], frames[0].shape[0]  # Get dimensions from first frame
# out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"MP4V"), 15, (w, 3 * h))
# for frame in frames:
#     out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
# out.release()
# print(f"✅ Video saved at {VIDEO_PATH}")

import os
import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

from ssd_model import HydraNet, SSD300MB9
from utils import prepare_img, drawboxes, depth_to_rgb

# -------- CONFIG --------
IMAGE_DIR = "data"
VIDEO_PATH = "output/ssd_output.mp4"
CKPT_PATH = "weights/ExpKITTI_joint.ckpt"
SSD_PATH = "weights/SSDMB9_001.tar"
CMAP_PATH = "cmap_kitti.npy"
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD COLORMAP --------
CMAP = np.load(CMAP_PATH)

# -------- MODEL SETUP --------
hydranet = HydraNet()
hydranet.define_mobilenet()
hydranet.define_lightweight_refinenet()
hydranet.object_detection_head = 1
hydranet.load_state_dict(torch.load(CKPT_PATH)["state_dict"])
hydranet.eval().to(DEVICE)

ssd_model = SSD300MB9(num_classes=9, hydra=hydranet)
ssd_model.load_state_dict(torch.load(SSD_PATH))
ssd_model.eval().to(DEVICE)

# -------- INFERENCE LOOP --------
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
result_video = []

for path in tqdm(image_paths, desc="Running SSD Inference"):
    img = Image.open(path).convert("RGB").resize((640, 192))
    img_np = np.array(img)
    w, h = img.size

    input_tensor = prepare_img(img_np).to(DEVICE)

    with torch.no_grad():
        segm, depth, locs_pred, cls_pred = ssd_model(input_tensor.unsqueeze(0))
        boxes, labels, scores = ssd_model.detect(locs_pred, cls_pred, min_score=0.4, max_overlap=0.5, top_k=200)
        boxes = boxes[0].cpu()
        labels = labels[0].cpu().tolist()
        scores = scores[0].cpu().tolist()

    segm = segm[0, :NUM_CLASSES].cpu().numpy().transpose(1, 2, 0)
    segm = cv2.resize(segm, (640, 192), interpolation=cv2.INTER_CUBIC)
    segm = CMAP[np.argmax(segm, axis=2)].astype(np.uint8)

    depth = depth[0, 0].cpu().numpy()
    depth = cv2.resize(depth, (640, 192), interpolation=cv2.INTER_CUBIC)
    depth_rgb = depth_to_rgb(depth)

    det_img = drawboxes(img_np.copy(), boxes, labels, scores)

    stacked = cv2.vconcat([img_np, segm, depth_rgb, det_img])
    result_video.append(stacked)

# -------- SAVE VIDEO --------
os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 4 * 192))
for frame in result_video:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print(f"✅ SSD video saved to {VIDEO_PATH}")
