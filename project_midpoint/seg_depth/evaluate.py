import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# === Segmentation Setup === #
MODEL_COLORS = [
    (128, 192, 128),  # road
    (192, 0, 128),    # car
    (128, 128, 0),    # vegetation
    (128, 0, 0),      # building
    (128, 128, 128),  # sky
    (0, 0, 64),       # sidewalk
    (64, 64, 128),    # fence
]
CLASS_NAMES = [
    "road",
    "car",
    "vegetation",
    "building",
    "sky",
    "sidewalk",
    "fence"
]
IGNORE_COLOR = (0, 0, 0)  # used for "ignore"
COLOR_TO_INDEX = {color: idx for idx, color in enumerate(MODEL_COLORS)}

def rgb_to_class_indices(rgb_img):
    h, w, _ = rgb_img.shape
    label_map = np.full((h, w), fill_value=255, dtype=np.uint8)  # 255 = ignore_index
    for rgb, idx in COLOR_TO_INDEX.items():
        mask = np.all(rgb_img == rgb, axis=-1)
        label_map[mask] = idx
    return label_map

def compute_segmentation_metrics(y_true, y_pred, num_classes):
    mask = (y_true != 255)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    acc = np.diag(cm).sum() / cm.sum()
    acc_cls = np.diag(cm) / cm.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
    mean_iu = np.nanmean(iu)
    freq = cm.sum(1) / cm.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Frequency Weighted IoU": fwavacc,
        "Class IoUs": iu,
    }

# === Depth Evaluation === #
def load_depth(path):
    depth = np.array(Image.open(path)).astype(np.float32)
    return depth / 256.0  # scale back to meters

def compute_depth_metrics(gt, pred):
    mask = (gt > 0) & (gt < 80)  # valid GT depth range

    gt = gt[mask]
    pred = pred[mask]

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    mae = np.mean(np.abs(gt - pred))

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    return {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'MAE': mae,
        'δ < 1.25': d1,
        'δ < 1.25²': d2,
        'δ < 1.25³': d3
    }

# === Combined Evaluation === #
def evaluate_all(seg_gt_dir, seg_pred_dir, depth_gt_dir, depth_pred_dir):
    # --- Segmentation ---
    seg_gt_paths = sorted(glob.glob(os.path.join(seg_gt_dir, "*.png")))
    seg_pred_paths = sorted(glob.glob(os.path.join(seg_pred_dir, "*.png")))
    assert len(seg_gt_paths) == len(seg_pred_paths), "Mismatch in number of segmentation images."

    all_gt, all_pred = [], []

    for gt_path, pred_path in tqdm(zip(seg_gt_paths, seg_pred_paths), total=len(seg_gt_paths), desc="Evaluating Segmentation"):
        gt_img = np.array(Image.open(gt_path))
        pred_img = np.array(Image.open(pred_path))

        gt_label = rgb_to_class_indices(gt_img)
        pred_label = rgb_to_class_indices(pred_img)

        all_gt.append(gt_label.flatten())
        all_pred.append(pred_label.flatten())

    y_true = np.concatenate(all_gt)
    y_pred = np.concatenate(all_pred)

    seg_metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=len(CLASS_NAMES))

    print("\nSegmentation Evaluation Metrics (Pretrained Model from https://arxiv.org/pdf/1809.04766):")
    for k, v in seg_metrics.items():
        if k == "Class IoUs":
            print("\nPer-Class IoU:")
            for i, val in enumerate(v):
                print(f"{CLASS_NAMES[i]:<10} IoU: {val:.4f}")
        else:
            print(f"{k}: {v:.4f}")

    # --- Depth ---
    depth_gt_paths = sorted(glob.glob(os.path.join(depth_gt_dir, "*.png")))
    depth_pred_paths = sorted(glob.glob(os.path.join(depth_pred_dir, "*.png")))
    assert len(depth_gt_paths) == len(depth_pred_paths), "Mismatch in number of depth images."

    depth_metrics = []

    for gt_path, pred_path in tqdm(zip(depth_gt_paths, depth_pred_paths), total=len(depth_gt_paths), desc="Evaluating Depth"):
        gt = load_depth(gt_path)
        pred = load_depth(pred_path)
        result = compute_depth_metrics(gt, pred)
        depth_metrics.append(result)

    keys = depth_metrics[0].keys()
    avg = {k: np.mean([m[k] for m in depth_metrics]) for k in keys}

    print("\nDepth Evaluation Metrics:")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    seg_gt_dir = "project_midpoint/seg_depth/ground_truth/seg_remapped"
    seg_pred_dir = "project_midpoint/seg_depth/predictions/seg"
    depth_gt_dir = "project_midpoint/seg_depth/ground_truth/depth"
    depth_pred_dir = "project_midpoint/seg_depth/predictions/depth"

    evaluate_all(seg_gt_dir, seg_pred_dir, depth_gt_dir, depth_pred_dir)
