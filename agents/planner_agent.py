# agents/planner_agent.py (Dynamic Version)

import json
import re # Add import for JSON extraction fallback
import google.generativeai as genai
# Import API key configuration
from src.config import GOOGLE_API_KEY 

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing for Planner Agent.")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash") # Use a Gemini model

def make_plan(image_path, options):
    """
    Generates a dynamic execution plan based on user options using an LLM.
    
    options: {
        "run_iou": bool,
        "show_boxes": bool,
        "auto_save": bool
    }
    """
    
    # Define available steps for the LLM to choose from
    all_steps = [
        "load_image",
        "run_perception_agent",
        "run_correction_agent",
        "convert_to_yolo"
    ]
    if options.get('run_iou', True):
        all_steps.append("evaluate_annotation")
    if options.get('show_boxes', True):
        all_steps.append("visualize_boxes")
    if options.get('auto_save', True):
        all_steps.append("save_results")

    # 1. Define the planning prompt
    prompt = f"""
    You are a high-level Planner Agent for an image annotation pipeline.
    Your task is to generate the optimal sequence of steps for the current run.
    
    The pipeline must always include: load_image, run_perception_agent, run_correction_agent, and convert_to_yolo.
    
    Available optional steps, determined by user settings:
    - evaluate_annotation (IoU)
    - visualize_boxes
    - save_results
    
    The final chronological plan should include the following steps (order them logically):
    {all_steps}

    Generate the plan array.

    Return ONLY JSON:
    {{
      "plan": ["step_1", "step_2", ...],
      "image": "{image_path}"
    }}
    """
    
    # 2. Call the LLM to generate the plan
    response = model.generate_content(prompt)

    raw_text = response.text

    # Basic JSON extraction fallback (as used in your other agents)
    try:
        plan_data = json.loads(raw_text)
        return json.dumps(plan_data)
    except Exception:
        # Try to extract JSON using regex fallback
        match = re.search(r"(\{[\s\S]*\})", raw_text)
        if match:
            return match.group(0)

    # Fallback plan in case the LLM fails
    return json.dumps({
        "plan": ["load_image", "run_perception_agent", "run_correction_agent", "evaluate_annotation", "convert_to_yolo", "save_results"],
        "image": image_path
    })