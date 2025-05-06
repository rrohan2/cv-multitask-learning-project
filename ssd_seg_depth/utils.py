import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as co

IMG_SCALE  = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im
    
def drawboxes(image, detect_boxes, detect_labels, detect_scores, label_color_map=None, suppress=None):
    """Draw bounding boxes with labels and confidence scores on the image."""
    try:
        draw = ImageDraw.Draw(image)
    except:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    if label_color_map is None:
        # Create default color map
        unique_labels = set(detect_labels)
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#888888", "#FFA500", "#800080"]
        label_color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    for i in range(detect_boxes.size(0)):
        if suppress is not None and detect_labels[i] in suppress:
            continue

        box_location = detect_boxes[i].tolist()
        label = detect_labels[i]
        score = detect_scores[i]

        draw.rectangle(xy=box_location, outline=label_color_map[label], width=2)
        text = f"{label} - {score * 100:.1f}%"
        text_size = draw.textsize(text, font=font)
        text_location = [box_location[0], box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0], box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[label])
        draw.text(xy=text_location, text=text, fill='white', font=font)

    return np.array(image)
