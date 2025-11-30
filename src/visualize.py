from PIL import Image, ImageDraw, ImageFont
import os

def draw_boxes(image_path, annotation_json):
    """
    annotation_json: { "objects": [ {"label": "cat", "bbox_norm": [x,y,w,h]}, ... ] }
    Converts normalized boxes (0–1) → pixel coordinates, then draws them.
    """

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    draw = ImageDraw.Draw(img)

    # Try to load font, fallback if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    objs = annotation_json.get("objects", [])

    for obj in objs:
        # Check for normalized bbox
        if "bbox_norm" not in obj:
            continue

        x_norm, y_norm, w_norm, h_norm = obj["bbox_norm"]

        # Convert normalized → pixel
        x = int(x_norm * W)
        y = int(y_norm * H)
        w = int(w_norm * W)
        h = int(h_norm * H)

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw label
        label = obj.get("label", "")
        draw.text((x1 + 4, max(0, y1 - 20)), label, fill="red", font=font)

    return img
