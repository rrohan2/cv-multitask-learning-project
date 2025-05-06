import supervision as sv
import cv2

# BOX_ANNOTATOR = sv.BoxAnnotator(
#     color=sv.ColorPalette(), 
#     thickness=2, 
#     text_thickness=1, 
#     text_scale=0.5,
#     text_padding=2
# )

# BOX_ANNOTATOR = sv.BoxAnnotator(
#     color=sv.ColorPalette(colors=["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]), 
#     thickness=2,
#     text_thickness=1,
#     text_scale=0.5,
#     text_padding=2
# )

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette(colors=["red", "green", "blue", "yellow"]),
    thickness=2
)


# def draw_detections(image, boxes, scores, class_ids, class_map):

#     detections = sv.Detections(
#         xyxy=boxes,
#         confidence=scores,
#         class_id=class_ids,
#         tracker_id=None
#     )

#     # labels = [
#     #     f"{class_map[class_id]} {confidence:0.2f}"
#     #     for _, confidence, class_id, _ in detections
#     # ]

#     labels = [
#         f"{class_map[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, _ in detections
#     ]

    
#     out_image = BOX_ANNOTATOR.annotate(
#         scene=image.copy(),
#         detections=detections,
#         labels=labels,
#         skip_label=False
#     )

#     return out_image

# def draw_detections(image, boxes, scores, class_ids, class_map):
#     import supervision as sv

#     detections = sv.Detections(
#         xyxy=boxes,
#         confidence=scores,
#         class_id=class_ids
#     )

#     try:
#         labels = [
#             f"{class_map[class_id]} {score:.2f}"
#             for class_id, score in zip(detections.class_id, detections.confidence)
#         ]
#     except Exception as e:
#         print("⚠️ Failed to build labels:", e)
#         labels = [f"{class_id}" for class_id in class_ids]

#     box_annotator = sv.BoxAnnotator()

#     out_image = box_annotator.annotate(
#         scene=image.copy(),
#         detections=detections,
#         labels=labels
#     )
#     return out_image

def draw_detections(image, boxes, scores, class_ids, class_map):
    import supervision as sv
    import numpy as np
    import cv2

    # Return original image if no detections
    if boxes is None or len(boxes) == 0:
        return image

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    # Create Detections object
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    # Draw bounding boxes
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)

    # Manually add class labels and scores
    for box, score, class_id in zip(boxes, scores, class_ids):
        label = f"{class_map.get(class_id, str(class_id))} {score:.2f}"
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

    return annotated
