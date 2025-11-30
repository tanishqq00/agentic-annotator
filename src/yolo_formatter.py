import json
import re
from PIL import Image

def extract_json(text):
    # Extract JSON-like text: { ... } OR [ ... ]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return match.group(0)
    return '{"objects": []}'


def normalize_annotation_structure(data):
    """
    Normalize ANY Gemini output to:
    {
        "objects": [ {"label": ..., "bbox": [...]}, ... ]
    }
    """

    
    if isinstance(data, dict) and "objects" in data:
        return data

    
    if isinstance(data, list):
        new_list = []
        for item in data:
            if isinstance(item, dict) and "bbox" in item:
                new_list.append(item)
            else:
                # ignore bad items
                continue
        return {"objects": new_list}

    
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list):
                new_list = []
                for item in val:
                    if isinstance(item, dict) and "bbox" in item:
                        new_list.append(item)
                return {"objects": new_list}

    #  else return empty
    return {"objects": []}




def convert_to_yolo(normalized_json_str, image_path):
    data = json.loads(normalized_json_str)

    img = Image.open(image_path)
    W, H = img.size

    yolo_lines = []

    for obj in data["objects"]:
        x_norm, y_norm, w_norm, h_norm = obj["bbox_norm"]

        # Convert normalized to pixel
        x = x_norm * W
        y = y_norm * H
        w = w_norm * W
        h = h_norm * H

        # YOLO values: cx, cy, w, h (all normalized)
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        ww = w / W
        hh = h / H

        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

    return "\n".join(yolo_lines)
