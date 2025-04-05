# Model:
I adopt yolov8m.pt as the pretrained model, which is a medium-sized model with a balance between speed and accuracy.

# Data Annotation:
Both Roboflow(https://roboflow.com/) and CVAT(https://www.cvat.ai/) can be an option.

# Dataset
KITTI online website(https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_prev_2.zip), which contains 22394 test images, 7481 train images and a size of 35GB.

Because of computational resource limits, use (https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip), which contains 7518 test images, 7481 train images and a size of 12GB. For midpoint checkin, only use first 1000 test images.


# 



# Future Work:
We need to train the model based on KITTI dataset and improve the model performance.

# References:
State-of-the-art computer vision model. YOLOv8. (n.d.). https://yolov8.com/ 

Ultralytics. (n.d.). Ultralytics/ultralytics: Ultralytics Yolo11 🚀. GitHub. https://github.com/ultralytics/ultralytics 

Train Yolov8 object detection on a custom dataset. (n.d.). YouTube. https://www.youtube.com/watch?v=m9fH9OWn8YM 