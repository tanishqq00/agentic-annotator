from PIL import Image
import google.generativeai as genai

model = genai.GenerativeModel("gemini-2.5-flash")

def correct_annotation(image_path, annotation_json):
    img = Image.open(image_path)

    prompt = """
You are a Correction Agent.

Your task:
Fix JSON and ensure the output contains NORMALIZED bounding boxes:
bbox_norm: [x_norm, y_norm, width_norm, height_norm]

Rules:
- Values must be between 0 and 1
- bbox_norm MUST exist
- If pixel bbox is detected, convert it to normalized

Return ONLY JSON:
{
  "objects": [
    { "label": "LABEL", "bbox_norm": [..] }
  ]
}
"""




    response = model.generate_content([prompt, img, annotation_json])
    return response.text
