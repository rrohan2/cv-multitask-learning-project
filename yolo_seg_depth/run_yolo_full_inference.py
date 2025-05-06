# import os
# import sys
# sys.path.append(os.path.abspath("."))

# import cv2
# import torch
# import ast
# import numpy as np
# import matplotlib.pyplot as plt
# from glob import glob
# from tqdm import tqdm

# from ons_scripts.dataset import collate_fn
# from ons_scripts.network.model import HydraNet, define_mobilenet, define_lightweight_refinenet
# from ons_scripts.network.detection_head import DetectionHead
# from ons_scripts.postprocess import PostProcess, draw_detections

# # ---------- Config ----------
# IMAGE_HEIGHT = 192
# IMAGE_WIDTH = 640
# VIDEO_PATH = "output/yolo_inference_video.mp4"
# INFERENCE_FOLDER = "inference_data"
# MEAN_STD_FILE = "outputs/mean_std_kitti_bdd100k.txt"
# DETECTION_WEIGHTS = "experiments/hydranet_od/hydranet_od_E97_L3.6737_VL3.8041.pth"
# ENCODER_DECODER_CKPT = "hydranets-data/ExpKITTI_joint.ckpt"
# NUM_CLASSES = 14
# CLASS_MAP = {
#     'Car': 0, 'Pedestrian': 1, 'Van': 2, 'Cyclist': 3, 'Truck': 4, 'Tram': 5, 'Person_sitting': 6,
#     "Rider": 7, "Bus": 8, "Train": 9, "Motorcycle": 10, "Bicycle": 11, "Traffic-sign": 12, "Traffic-light": 13
# }
# CLASS_ID_TO_NAME_MAPPING = {v: k for k, v in CLASS_MAP.items()}

# # ---------- Setup ----------
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open(MEAN_STD_FILE, "r") as f:
#     lines = f.readlines()
# MEAN = ast.literal_eval(" ".join(lines[0].split()[1:]))
# STD = ast.literal_eval(" ".join(lines[1].split()[1:]))

# postprocess = PostProcess(IMAGE_HEIGHT, IMAGE_WIDTH, conf_thres=0.3, iou_thres=0.3)

# model = HydraNet()
# model.define_mobilenet()
# model.define_lightweight_refinenet()
# model.load_state_dict(torch.load(ENCODER_DECODER_CKPT)["state_dict"])
# model.extend_object_detection = True
# model.only_return_object_detection_result = False
# model.detect_head = DetectionHead(
#     num_classes=NUM_CLASSES,
#     decoder_channels=(256, 256, 256),
#     head_channels=(64, 128, 256),
#     stride=(8, 16, 32),
#     reg_max=16
# )
# model.detect_head.load_state_dict(torch.load(DETECTION_WEIGHTS))
# model.to(DEVICE)
# model.eval()

# # ---------- Utils ----------
# def read_image(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
#     return img

# def preprocess_image(image):
#     image = image / 255.0
#     image = (image - np.array(MEAN).reshape((1, 1, 3))) / np.array(STD).reshape((1, 1, 3))
#     image = np.moveaxis(image.astype(np.float32), -1, 0)
#     return torch.from_numpy(image).unsqueeze(0)

# def depth_to_rgb(depth):
#     depth = np.abs(depth)
#     depth = np.clip(depth / np.max(depth), 0, 1)
#     return (plt.cm.plasma(depth)[..., :3] * 255).astype(np.uint8)

# # ---------- Inference Loop ----------
# os.makedirs("output/inference_frames", exist_ok=True)
# image_paths = sorted(glob(os.path.join(INFERENCE_FOLDER, "*")))
# frames = []

# for idx, path in enumerate(tqdm(image_paths, desc="Running Inference")):
#     orig = read_image(path)
#     image_tensor = preprocess_image(orig).to(DEVICE)

#     with torch.no_grad():
#         segm_out, depth_out, det_out = model(image_tensor)
#         # boxes, scores, class_ids = postprocess(det_out)
#         if isinstance(det_out, tuple):
#             det_out = det_out[0]  # keep only the YOLO output tensor for inference
#         boxes, scores, class_ids = postprocess(det_out)

#     # Segmentation
#     segm_np = segm_out[0].argmax(dim=0).cpu().numpy()
#     segm_color = plt.cm.tab20(segm_np / NUM_CLASSES)[..., :3]
#     segm_color = (segm_color * 255).astype(np.uint8)
#     segm_color = cv2.resize(segm_color, (IMAGE_WIDTH, IMAGE_HEIGHT))

#     # Depth
#     depth_np = depth_out[0, 0].cpu().numpy()
#     depth_rgb = depth_to_rgb(cv2.resize(depth_np, (IMAGE_WIDTH, IMAGE_HEIGHT)))

#     # Draw detections
#     det_image = draw_detections(orig.copy(), boxes, scores, class_ids, CLASS_ID_TO_NAME_MAPPING)

#     stacked = np.vstack([orig, segm_color, depth_rgb, det_image])
#     frames.append(stacked)
#     cv2.imwrite(f"output/inference_frames/frame_{idx:04d}.jpg", cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

# # ---------- Write Video ----------
# h, w = frames[0].shape[:2]
# video_writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
# for f in frames:
#     video_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
# video_writer.release()

# print(f"\n✅ Video saved at: {VIDEO_PATH}")


import os
import sys
sys.path.append(os.path.abspath("."))

import cv2
import torch
import ast
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image

from ons_scripts.network.model import HydraNet, define_mobilenet, define_lightweight_refinenet
from ons_scripts.network.detection_head import DetectionHead
from ons_scripts.postprocess import PostProcess, draw_detections
from ons_scripts.utils import prepare_img, depth_to_rgb
# from multitask_project.utils import prepare_img, depth_to_rgb

# ---------- CONFIG ----------
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640
VIDEO_PATH = "outputs/videos/yolo_full_output_3.mp4"
INFERENCE_FOLDER = "inference_data"
MEAN_STD_FILE = "outputs/mean_std_kitti_bdd100k.txt"
DETECTION_WEIGHTS = "experiments/hydranet_od/hydranet_od_E97_L3.6737_VL3.8041.pth"
ENCODER_DECODER_CKPT = "hydranets-data/ExpKITTI_joint.ckpt"
CMAP_FILE = "data/cmap_kitti.npy"

NUM_CLASSES = 14
CLASS_MAP = {
    'Car': 0, 'Pedestrian': 1, 'Van': 2, 'Cyclist': 3, 'Truck': 4, 'Tram': 5, 'Person_sitting': 6,
    "Rider": 7, "Bus": 8, "Train": 9, "Motorcycle": 10, "Bicycle": 11, "Traffic-sign": 12, "Traffic-light": 13
}
CLASS_ID_TO_NAME_MAPPING = {v: k for k, v in CLASS_MAP.items()}

# ---------- SETUP ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CMAP = np.load(CMAP_FILE)

with open(MEAN_STD_FILE, "r") as f:
    lines = f.readlines()
MEAN = ast.literal_eval(" ".join(lines[0].split()[1:]))
STD = ast.literal_eval(" ".join(lines[1].split()[1:]))

model = HydraNet()
model.define_mobilenet()
model.define_lightweight_refinenet()
model.load_state_dict(torch.load(ENCODER_DECODER_CKPT)["state_dict"])
model.extend_object_detection = True
model.only_return_object_detection_result = False
model.detect_head = DetectionHead(
    num_classes=NUM_CLASSES,
    decoder_channels=(256, 256, 256),
    head_channels=(64, 128, 256),
    stride=(8, 16, 32),
    reg_max=16
)
model.detect_head.load_state_dict(torch.load(DETECTION_WEIGHTS))
model.to(DEVICE)
model.eval()

postprocess = PostProcess(IMAGE_HEIGHT, IMAGE_WIDTH, conf_thres=0.3, iou_thres=0.3)

# ---------- UTILS ----------
def apply_cmap(segm):
    return (CMAP[segm] * 255).astype(np.uint8)

def preprocess(image):
    image = image / 255.0
    image = (image - np.array(MEAN).reshape((1, 1, 3))) / np.array(STD).reshape((1, 1, 3))
    image = np.moveaxis(image.astype(np.float32), -1, 0)
    return torch.from_numpy(image).unsqueeze(0)

# ---------- INFERENCE ----------
image_paths = sorted(glob(os.path.join(INFERENCE_FOLDER, "*")))
result_video = []

for img_path in tqdm(image_paths, desc="Running Inference"):
    # image = np.array(Image.open(img_path).convert("RGB"))
    # h, w, _ = image.shape
    image = np.array(Image.open(img_path).convert("RGB").resize((640, 192)))
    h, w = 192, 640  # force h and w to match resized


    image_tensor = preprocess(image).to(DEVICE)

    with torch.no_grad():
        segm, depth, det_out = model(image_tensor)
        # boxes, scores, class_ids = postprocess(det_out)
        if isinstance(det_out, tuple):
            det_out = det_out[0]  # keep only the YOLO output tensor for inference
        boxes, scores, class_ids = postprocess(det_out)

    # Segmentation
    segm = segm[0, :NUM_CLASSES].cpu().numpy().transpose(1, 2, 0)
    segm = cv2.resize(segm, (w, h), interpolation=cv2.INTER_CUBIC)
    segm = apply_cmap(np.argmax(segm, axis=2))

    # Depth
    depth = depth[0, 0].cpu().numpy()
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    depth_rgb = depth_to_rgb(depth)

    # Detection
    det_image = draw_detections(image.copy(), boxes, scores, class_ids, CLASS_ID_TO_NAME_MAPPING)

    # Stack and save
    combined = cv2.vconcat([image, segm, depth_rgb, det_image])
    result_video.append(combined)

# ---------- SAVE VIDEO ----------
os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
out = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, 4 * h))
for frame in result_video:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print(f"✅ Saved video to {VIDEO_PATH}")
