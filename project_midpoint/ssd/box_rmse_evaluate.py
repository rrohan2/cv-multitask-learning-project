import os
import numpy as np
import argparse
from collections import defaultdict
import time

def parse_bbox_file(file_path, confidence_threshold=0.0):
    boxes = []
    scores = []
    classes = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:  
                    continue
                    
                class_name = parts[0]
                if class_name == "DontCare":
                    continue
                    
                try:
                    x_min = float(parts[4])
                    y_min = float(parts[5])
                    x_max = float(parts[6])
                    y_max = float(parts[7])
                    
                    confidence = float(parts[14])
                    
                    if confidence >= confidence_threshold:
                        boxes.append([x_min, y_min, x_max, y_max])
                        scores.append(confidence)
                        classes.append(class_name)
                except (ValueError, IndexError) as e:
                    pass
    except Exception as e:
        print(f"Wrong file {file_path} : {e}")
    
    return boxes, scores, classes

def calculate_iou(box1, box2):

    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    if x_min >= x_max or y_min >= y_max:
        return 0.0
    intersection = (x_max - x_min) * (y_max - y_min)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_box_rmse(pred_box, gt_box):
    squared_errors = [
        (pred_box[0] - gt_box[0])**2,  # xmin error
        (pred_box[1] - gt_box[1])**2,  # ymin error
        (pred_box[2] - gt_box[2])**2,  # xmax error
        (pred_box[3] - gt_box[3])**2   # ymax error
    ]
    
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

def calculate_coordinate_errors(pred_box, gt_box):
    return {
        'xmin': (pred_box[0] - gt_box[0]),
        'ymin': (pred_box[1] - gt_box[1]),
        'xmax': (pred_box[2] - gt_box[2]),
        'ymax': (pred_box[3] - gt_box[3])
    }

def calculate_coordinate_rmse(errors_list):
    result = {}
    for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
        values = [e[coord] for e in errors_list]
        rmse = np.sqrt(np.mean(np.array(values) ** 2))
        result[coord] = rmse
    return result

def calculate_normalized_rmse(pred_box, gt_box):
    width = gt_box[2] - gt_box[0]
    height = gt_box[3] - gt_box[1]
    size = np.sqrt(width**2 + height**2)
    
    # Calculate RMSE
    rmse = calculate_box_rmse(pred_box, gt_box)
    
    if size > 1.0:
        return rmse / size
    else:
        return rmse

def simple_matching(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold, ignore_class=True):
    matched_pairs = []
    matched_gt = [False] * len(gt_boxes)
    
    for i, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1
        
        for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if not matched_gt[j]:
                if ignore_class or pred_class == gt_class:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt[best_gt_idx] = True
            matched_pairs.append((i, best_gt_idx, best_iou))
    
    return matched_pairs

def evaluate_box_rmse(gt_dir, pred_dir, iou_threshold=0.5, confidence_threshold=0.0, ignore_class=True, verbose=False):
    """Evaluate the RMSE error of the bounding box"""
    # Get all file names
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt') and f != 'README.md']
    
    # Data statistics
    all_rmse_errors = []                # All RMSE error
    all_normalized_rmse_errors = []     # All normalized RMSE error
    class_rmse_errors = defaultdict(list)  # RMSE error of each class
    class_coord_errors = defaultdict(list)  # Coordinate error of each class
    class_matches = defaultdict(int)       # Number of matches of each class
    
    # Coordinate error
    all_coord_errors = []
    
    # Confidence error statistics
    all_confidence_errors = []
    
    print(f"Start RMSE evaluation, found {len(gt_files)} test samples...")
    
    start_time = time.time()
    processed_files = 0
    matched_pairs = 0
    class_counts = defaultdict(int)  # For statistics of the number of true boxes of each class
    
    # Progress report
    print_interval = max(1, len(gt_files) // 20)  # Report progress every 5%
    
    for file_idx, gt_file in enumerate(gt_files):
        if file_idx % print_interval == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (file_idx + 1)) * (len(gt_files) - file_idx - 1) if file_idx > 0 else 0
            print(f"Processing: {file_idx}/{len(gt_files)} ({file_idx/len(gt_files)*100:.1f}%) ETA: {eta:.0f}s")
        
        base_name = os.path.splitext(gt_file)[0]
        pred_file = os.path.join(pred_dir, f"{base_name}.txt")
        
        # Check if the prediction file exists
        if not os.path.exists(pred_file):
            continue
        
        # Parse the ground truth and prediction
        gt_boxes, gt_scores, gt_classes = parse_bbox_file(os.path.join(gt_dir, gt_file))
        pred_boxes, pred_scores, pred_classes = parse_bbox_file(pred_file, confidence_threshold)
        
        # Count the number of classes
        for cls in gt_classes:
            class_counts[cls] += 1
        
        # If there is no ground truth or prediction, skip this file
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
            
        processed_files += 1
        
        # Match the ground truth and prediction
        matches = simple_matching(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold, ignore_class)
        matched_pairs += len(matches)
        
        for pred_idx, gt_idx, iou in matches:
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            pred_class = pred_classes[pred_idx]
            pred_score = pred_scores[pred_idx]
            
            # Calculate the RMSE error of the bounding box
            rmse_error = calculate_box_rmse(pred_box, gt_box)
            all_rmse_errors.append(rmse_error)
            
            # Calculate the normalized RMSE error
            norm_rmse_error = calculate_normalized_rmse(pred_box, gt_box)
            all_normalized_rmse_errors.append(norm_rmse_error)
            
            # Count the RMSE error by class
            class_rmse_errors[pred_class].append(rmse_error)
            class_matches[pred_class] += 1
            
            # Calculate the error of each coordinate
            coord_errors = calculate_coordinate_errors(pred_box, gt_box)
            all_coord_errors.append(coord_errors)
            class_coord_errors[pred_class].append(coord_errors)
            
            # If the ground truth has confidence, calculate the confidence error
            if len(gt_scores) > gt_idx:
                conf_error = abs(pred_score - gt_scores[gt_idx])
                all_confidence_errors.append(conf_error)
    
    total_time = time.time() - start_time
    print(f"Evaluation completed, time: {total_time:.2f}s")
    
    # Print category statistics information
    print("\nNumber of true boxes of each class:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count}")
    
    # Calculate the overall RMSE
    if all_rmse_errors:
        avg_rmse_error = np.mean(all_rmse_errors)
        median_rmse_error = np.median(all_rmse_errors)
        
        avg_norm_rmse_error = np.mean(all_normalized_rmse_errors)
        median_norm_rmse_error = np.median(all_normalized_rmse_errors)
        
        # Calculate the RMSE of each coordinate
        coord_rmse = calculate_coordinate_rmse(all_coord_errors)
        
        # Confidence average error
        avg_conf_error = np.mean(all_confidence_errors) if all_confidence_errors else "N/A"
        
        # Print the evaluation results
        print(f"\n===== RMSE Evaluation (IoU={iou_threshold}) =====")
        print(f"Processed files: {processed_files}/{len(gt_files)}")
        print(f"Matching bounding box pairs: {matched_pairs}")
        print(f"Average RMSE error: {avg_rmse_error:.2f} pixels")
        print(f"Median RMSE error: {median_rmse_error:.2f} pixels")
        print(f"Average normalized RMSE error: {avg_norm_rmse_error:.4f}")
        print(f"Median normalized RMSE error: {median_norm_rmse_error:.4f}")
        
        print("\nCoordinate RMSE:")
        for coord, error in coord_rmse.items():
            print(f"  {coord}: {error:.2f} pixels")
        
        if avg_conf_error != "N/A":
            print(f"\nConfidence average error: {avg_conf_error:.4f}")
        
        # Print the RMSE error of each class
        print("\n===== RMSE Error of Each Class =====")
        for cls, errors in sorted(class_rmse_errors.items(), key=lambda x: len(x[1]), reverse=True):
            if errors:
                avg_error = np.mean(errors)
                median_error = np.median(errors) if len(errors) > 1 else avg_error
                
                # Calculate the RMSE of each coordinate of this class
                cls_coord_rmse = calculate_coordinate_rmse(class_coord_errors[cls])
                
                print(f"{cls}:")
                print(f"  Number of samples: {len(errors)}")
                print(f"  Average RMSE error: {avg_error:.2f} pixels")
                print(f"  Median RMSE error: {median_error:.2f} pixels")
                print(f"  Coordinate RMSE: xmin={cls_coord_rmse['xmin']:.2f}, ymin={cls_coord_rmse['ymin']:.2f}, xmax={cls_coord_rmse['xmax']:.2f}, ymax={cls_coord_rmse['ymax']:.2f}")
                
        return {
            'avg_rmse_error': avg_rmse_error,
            'median_rmse_error': median_rmse_error,
            'avg_norm_rmse_error': avg_norm_rmse_error,
            'matched_pairs': matched_pairs,
            'class_matches': class_matches,
            'coord_rmse': coord_rmse
        }
    else:
        print("\nNo matching bounding boxes found, cannot calculate RMSE error.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI bounding box RMSE error evaluation")
    parser.add_argument("--gt_dir", type=str, help="Ground truth directory")
    parser.add_argument("--pred_dir", type=str, help="Prediction directory")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="IoU threshold")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--ignore_class", action="store_true", help="Whether to ignore class differences")
    parser.add_argument("--verbose", action="store_true", help="Whether to output detailed information")
    
    args = parser.parse_args()
    
    # If no parameters are provided, use the default directory
    if args.gt_dir is None or args.pred_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.gt_dir = args.gt_dir or os.path.join(current_dir, "ground_truth")
        args.pred_dir = args.pred_dir or os.path.join(current_dir, "predictions")
    
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Ignore class differences: {args.ignore_class}")
    
    # Evaluate
    evaluate_box_rmse(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        ignore_class=args.ignore_class,
        verbose=args.verbose
    ) 