import sys
import os
sys.path.append(os.path.abspath("."))
import numpy as np
from PIL import Image
import os
import glob

KITTI_TO_MODEL = {
    (128, 64, 128): (128, 192, 128),   # road
    (0, 0, 142): (192, 0, 128),        # car
    (107, 142, 35): (128, 128, 0),     # vegetation
    (70, 70, 70): (128, 0, 0),         # building
    (70, 130, 180): (128, 128, 128),   # sky
    (244, 35, 232): (0, 0, 64),        # sidewalk
    (190, 153, 153): (64, 64, 128),    # fence
}

IGNORE_COLOR = (0, 0, 0)  # Or (255, 255, 255) if you prefer

def remap_gt_ignore_unknowns(gt_path, output_path):
    img = np.array(Image.open(gt_path))
    reshaped = img.reshape(-1, 3)

    remapped = np.array([
        KITTI_TO_MODEL.get(tuple(pixel), IGNORE_COLOR)
        for pixel in reshaped
    ], dtype=np.uint8).reshape(img.shape)

    Image.fromarray(remapped).save(output_path)

# Apply to all images
if __name__ == "__main__":
    input_dir = "project_midpoint/seg_depth/ground_truth/seg"
    output_dir = "project_midpoint/seg_depth/ground_truth/seg_remapped"
    os.makedirs(output_dir, exist_ok=True)

    for path in glob.glob(os.path.join(input_dir, "*.png")):
        fname = os.path.basename(path)
        remap_gt_ignore_unknowns(path, os.path.join(output_dir, fname))

    print("KITTI ground truth remapped with ignore mask for unknown classes.")
