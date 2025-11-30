# planner_agent.py
import json

def make_plan(image_path):
    
    plan = {
        "plan": [
            "load_image",
            "run_perception_agent",
            "run_correction_agent",
            "evaluate_annotation",
            "convert_to_yolo",
            "save_results"
        ],
        "image": image_path
    }
    return json.dumps(plan)
