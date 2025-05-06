import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

class BDD100KDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        transform,
        class_map,
        image_height,
        image_width,
        mean=[0., 0., 0.],
        std=[1., 1., 1.]
    ):
        
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.class_map = class_map
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

        # Extract class indices using class_map
        class_ids = np.array([self.class_map[class_name] for class_name in arr[:, 0]], dtype=int)

        # Extract xyxy coordinates
        bboxes = arr[:, 2:6].astype(float)
                                 
        return class_ids.tolist(), bboxes.tolist()
    
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
            # if file is empty, get random image and labels
            random_idx = np.random.randint(0, len(self.inputs))
            return self.__getitem__(random_idx)
        
        # get class_ids and bboxes
        class_ids, bboxes = self.get_class_ids_and_bboxes(labels)

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

def collate_fn(batch):
    """
    Collates data samples into batches.
    
    Note:
    Output dimension (along 0th axis) of `stacked_images`
    may or may not be equal to the `concatenated_targets`.
    
    """
    images, targets = zip(*batch)

    # Stack images along the new dimension 0; shape = (B, C, H, W)
    stacked_images = torch.stack(images, dim=0)
    
    # Concatenate the tensors along newly created dimension 0
    # which represents the batch index; shape = (Num_batch_Boxes, 6)
    # [batch_id, class_index, xn, yn, wn, hn]
    concatenated_targets = torch.cat([
        torch.cat([idx * torch.ones(target.size(0), 1), target], dim=1)
        for idx, target in enumerate(targets)
    ], dim=0)

    return stacked_images, concatenated_targets