import os
import torch
import cv2
import numpy as np
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F

# KITTI dataset classes (mapped from corresponding COCO categories)
KITTI_CLASSES = ["Background", "Car", "Pedestrian", "Cyclist"]
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Mapping from COCO classes to KITTI classes
COCO_TO_KITTI = {
    'person': 'Pedestrian',
    'bicycle': 'Cyclist',
    'car': 'Car',
    'motorcycle': 'Cyclist',  # Motorcycles are also categorized as Cyclist
    'bus': 'Van',  
    'truck': 'Truck'  
}

def load_model(device='cuda'):
    """Load pre-trained SSD model"""
    # Use pre-trained SSD300 model (based on COCO dataset)
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.to(device).eval()
    print(f"Loaded pre-trained SSD300 model (based on COCO dataset, {len(COCO_CLASSES)} classes)")
    return model

def preprocess_image(image_path, target_size=300):
    """Image preprocessing"""
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default, convert to RGB
    image = F.to_tensor(image)  # Convert to Tensor and normalize to [0,1]
    image = F.resize(image, [target_size, target_size])
    return image.unsqueeze(0), (original_width, original_height)  # Add batch dimension and return original size

def coco_class_to_kitti(coco_idx):
    """Convert COCO class index to KITTI class name"""
    if coco_idx < 1 or coco_idx >= len(COCO_CLASSES):
        return "DontCare"  # For unrecognized classes, use KITTI's "DontCare" label
    
    coco_class_name = COCO_CLASSES[coco_idx]
    return COCO_TO_KITTI.get(coco_class_name, "DontCare")  # Default to DontCare

def rescale_boxes(boxes, model_size, original_size):
    """Rescale boxes from model input size to original image size"""
    w_ratio = original_size[0] / model_size
    h_ratio = original_size[1] / model_size
    
    scaled_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        scaled_boxes.append([
            xmin * w_ratio,
            ymin * h_ratio,
            xmax * w_ratio,
            ymax * h_ratio
        ])
    
    return np.array(scaled_boxes)

def save_predictions(boxes, scores, labels, output_txt_path, confidence_threshold=0.2):
    """Save predictions in KITTI format (.txt)"""
    with open(output_txt_path, 'w') as f:
        for box, score, label_idx in zip(boxes, scores, labels):
            if score < confidence_threshold:
                continue
                
            # Convert to KITTI format coordinates
            xmin, ymin, xmax, ymax = map(float, box)
            
            # Convert COCO class ID to KITTI class name
            kitti_class = coco_class_to_kitti(label_idx)
            
            # Only save classes we care about (cars, pedestrians, etc.)
            if kitti_class not in ["Car", "Pedestrian", "Cyclist", "Van", "Truck"]:
                continue
                
            # Save prediction results in KITTI format
            f.write(f"{kitti_class} -1 -1 -10 {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f} -1 -1 -1 -1000 -1000 -1000 0 {score:.2f}\n")

def run_inference(model, image_dir, output_dir, confidence_threshold=0.2, target_size=300):
    """Run batch inference and save results"""
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Found {total_images} image files")
    
    for img_name in image_files:
        # 1. Preprocessing
        base_name = os.path.splitext(img_name)[0]
        image_path = os.path.join(image_dir, img_name)
        
        try:
            # Process image
            image_tensor, original_size = preprocess_image(image_path, target_size)
            image_tensor = image_tensor.to(device)
            
            # 2. Inference
            with torch.no_grad():
                predictions = model(image_tensor)
            
            # 3. Post-processing
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Scale boxes back to original image size
            boxes = rescale_boxes(boxes, target_size, original_size)
            
            # 4. Save prediction results
            output_txt_path = os.path.join(output_dir, f"{base_name}.txt")
            save_predictions(boxes, scores, labels, output_txt_path, confidence_threshold)
            processed_count += 1
            
            # Print progress
            if processed_count % 50 == 0 or processed_count == total_images:
                print(f"Progress: {processed_count}/{total_images} ({processed_count/total_images*100:.1f}%)")
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SSD object detection inference")
    parser.add_argument("--data_dir", type=str, help="Input image directory")
    parser.add_argument("--output_dir", type=str, help="Output prediction directory")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--target_size", type=int, default=300, help="Target size for SSD model")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(device)
    
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(current_dir, "data")
    output_dir = args.output_dir or os.path.join(current_dir, "predictions")
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Target size: {args.target_size}")
    
    run_inference(
        model,
        image_dir=data_dir,            
        output_dir=output_dir,
        confidence_threshold=args.confidence,
        target_size=args.target_size
    )