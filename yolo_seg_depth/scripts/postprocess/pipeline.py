import cv2
import numpy as np
from .nms import multiclass_nms
from .utils import xywh2xyxy

class PostProcess:
    def __init__(
        self,
        image_height,
        image_width,
        conf_thres=0.3,
        iou_thres=0.3
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    
    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        return boxes

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def __call__(self, outputs):
        
        boxes_list = []
        scores_list = []
        class_ids_list = []

        outputs = outputs.detach().cpu().numpy()    # shape: (batch_size, 18, 2520)
        batch_size = outputs.shape[0]

        for output in outputs:
            predictions = output.T        # shape: (2520, 18)

            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > self.conf_thres, :]
            scores = scores[scores > self.conf_thres]

            if len(scores) == 0:
                boxes_list.append([])
                scores_list.append([])
                class_ids_list.append([])
                continue

            # Get the class with the highest confidence
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Get bounding boxes for each object
            boxes = self.extract_boxes(predictions)

            # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
            # indices = nms(boxes, scores, self.iou_thres)
            indices = multiclass_nms(boxes, scores, class_ids, self.iou_thres)

            boxes_list.append(boxes[indices])
            scores_list.append(scores[indices])
            class_ids_list.append(class_ids[indices])


        if batch_size == 1:
            return boxes_list[0], scores_list[0], class_ids_list[0]

        return boxes_list, scores_list, class_ids_list