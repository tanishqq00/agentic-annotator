import google.generativeai as genai
from PIL import Image
import json


prompt = """
You are an Image Annotation Agent.

Your task:
Detect objects in the image and output NORMALIZED bounding boxes.

Normalized bounding boxes:
- bbox_norm = [x_norm, y_norm, width_norm, height_norm]
- All values are floats between 0 and 1
- x_norm = x_pixel / image_width
- y_norm = y_pixel / image_height

Important:
- DO NOT return pixel values.
- DO NOT return code blocks or markdown.
- Return ONLY JSON.

Correct format:
{
  "objects": [
    {
      "label": "LABEL",
      "bbox_norm": [x_norm, y_norm, width_norm, height_norm]
    }
  ]
}
"""

# Main function your main.py will call
def annotate_image(image_path):
    img = Image.open(image_path)

    # Send prompt + image to LLM
    model = genai.GenerativeModel("gemini-2.5-flash")  # or 1.5-pro
    response = model.generate_content(
        [prompt, img],
        stream=False,
    )

    raw_text = response.text

    # Ensure we extract only JSON
    try:
        # If model returned a JSON chunk directly
        data = json.loads(raw_text)
        return json.dumps(data)
    except Exception:
        # Try to extract JSON using regex fallback
        import re
        match = re.search(r"(\{[\s\S]*\})", raw_text)
        if match:
            return match.group(0)

    # Fallback (rare)
    return '{"objects":[]}'

