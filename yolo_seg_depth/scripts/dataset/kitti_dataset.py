import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        transform,
        class_map,
        ignore_classes,
        image_height,
        image_width,
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
    ):
        
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.class_map = class_map
        self.ignore_classes = ignore_classes
        self.image_height = image_height
        self.image_width = image_width
        
        self.final_transform = A.Compose([
                A.Normalize(mean=mean, std=std, max_pixel_value=255),
                ToTensorV2(),
        ])
    
    def read_image(self, path):
        return cv2.cvtColor(
            cv2.imread(path, cv2.IMREAD_COLOR), 
            cv2.COLOR_BGR2RGB
        )
    
    def read_label(self, path):
        with open(path, 'r') as f:
            labels = f.read().splitlines()
        return labels
    
    def get_class_ids_and_bboxes(self, labels):
        # Convert the list to a NumPy array
        arr = np.array([line.split() for line in labels])

        # Filter out rows with "DontCare" or "Misc"
        mask = ~np.isin(arr[:, 0], self.ignore_classes)
        filtered_arr = arr[mask]

        # Extract class indices using class_map
        class_ids = np.array([self.class_map[class_name] for class_name in filtered_arr[:, 0]], dtype=int)

        # Extract xyxy coordinates
        bboxes = filtered_arr[:, 4:8].astype(float)
                                 
        return bboxes.tolist(), class_ids.tolist()
    
    def pascal_voc_to_yolo(self, bboxes):
        
        for i in range(len(bboxes)):
            xmin, ymin, xmax, ymax = bboxes[i]

            # Calculate center coordinates
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin

            # Normalize coordinates and dimensions
            x_center /= self.image_width
            y_center /= self.image_height
            width /= self.image_width
            height /= self.image_height

            bboxes[i] = [x_center, y_center, width, height]
                
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        # Load image and label
        image = self.read_image(self.inputs[idx])
        labels = self.read_label(self.targets[idx])

        if not labels:
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)
        
        # get class_ids and bboxes
        bboxes, class_ids = self.get_class_ids_and_bboxes(labels)

        if not bboxes:
            # if image has 0 bboxes, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)

        # preprocess
        aug = self.transform(image=image, bboxes=bboxes, category_ids=class_ids)
        image = aug["image"]
        bboxes = aug["bboxes"]
        class_ids = aug["category_ids"]
        
        if not bboxes:
            # after preprocess; if image has 0 bboxes, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)
        
        # final transform
        image = self.final_transform(image=image)["image"]

        # convert pascal_voc bboxes (xyxy) to yolo bboxes (xn,yn,wn,hn : normalized)
        self.pascal_voc_to_yolo(bboxes)
        
        # Combine class indices and bboxes coordinates
        target = torch.column_stack([
            torch.tensor(class_ids, dtype=torch.float32),
            torch.tensor(bboxes, dtype=torch.float32)
        ]) # shape: (num_bboxes, 5); [class_index, xn, yn, wn, hn]

        return image, target
