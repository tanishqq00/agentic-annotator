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




# tanishqq00/agentic-annotator/agentic-annotator-279ab10d37512c5a77327e16f824b84faa7b35ec/src/yolo_formatter.py

...
def convert_to_yolo(normalized_json_str, image_path):
    data = json.loads(normalized_json_str)

    # Note: W, H are only needed here for image opening, but not for calculation below 
    # since we work with normalized coordinates directly.
    img = Image.open(image_path)
    # W, H = img.size # No longer needed for calculation

    yolo_lines = []

    for obj in data["objects"]:
        x_norm, y_norm, w_norm, h_norm = obj["bbox_norm"]

        # Convert normalized top-left (x_norm, y_norm, w_norm, h_norm) 
        # to normalized center (cx, cy, ww, hh) directly.
        
        cx = x_norm + (w_norm / 2) # Normalized center x
        cy = y_norm + (h_norm / 2) # Normalized center y
        ww = w_norm                # Normalized width
        hh = h_norm                # Normalized height

        # The previous pixel conversions are now simplified:
        # x = x_norm * W
        # y = y_norm * H
        # w = w_norm * W
        # h = h_norm * H
        # cx = (x + w / 2) / W 
        #    = ((x_norm * W) + (w_norm * W) / 2) / W
        #    = x_norm + w_norm / 2  <-- New simplified formula

        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

    return "\n".join(yolo_lines)
