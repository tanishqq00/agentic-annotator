
import json
import logging
from datetime import datetime
from PIL import Image

from src.config import GOOGLE_API_KEY
import google.generativeai as genai

# Agents & utilities
from agents.perception_agent import annotate_image
from agents.correction_agent import correct_annotation
from agents.planner_agent import make_plan

from src.yolo_formatter import convert_to_yolo
from src.tools import save_text
from src.session_service import InMemorySessionService
from src.memory_bank import remember, recall
from src.eval import iou

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


#   0. Configure LLM

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing.")

genai.configure(api_key=GOOGLE_API_KEY)
logger.info("API key loaded ✔")


IMAGE_PATH = "data/sample.jpeg"
GT_PATH = "annotations/gt_sample.json"
RAW_OUT = "annotations/raw.json"
CORR_OUT = "annotations/final.json"
YOLO_OUT = "annotations/final.txt"



# 1. Create Session

session = InMemorySessionService()
sid = session.create_session(user="pipeline_runner")
session.add_event(sid, "Session started")
logger.info("Session created.")



# 2. PLANNING (Agent #1)

# main.py (Around line 40)

...
# 2. PLANNING (Agent #1)

# Define default options for the pipeline run
default_options = {
    "run_iou": True,
    "show_boxes": False, # Visualization is not typically run in the console script
    "auto_save": True
}

try:
    # Pass the image path and the default options to the dynamic Planner Agent
    plan_json = make_plan(IMAGE_PATH, default_options)
    plan = json.loads(plan_json)
    session.add_event(sid, f"Plan created.")
    logger.info("Planning completed.")
    logger.debug(plan_json)
except Exception as e:
    logger.error(f"Planner agent failed: {e}")
    raise



# 3. PERCEPTION (Agent #2)

try:
    session.add_event(sid, "Running perception agent…")
    raw = annotate_image(IMAGE_PATH)
    save_text(RAW_OUT, raw)
    session.add_event(sid, "Perception agent completed.")
    logger.info("Perception agent output saved.")
except Exception as e:
    logger.error(f"Perception agent failed: {e}")
    raise


# 4. CORRECTION (Agent #3)

try:
    session.add_event(sid, "Running correction agent…")
    corrected = correct_annotation(IMAGE_PATH, raw)


    try:
        corrected_json = json.loads(corrected)
    except Exception:
        import re
        match = re.search(r"(\{[\s\S]*\})", corrected)
        if match:
            corrected_json = json.loads(match.group(0))
        else:
            corrected_json = {"objects": []}

    save_text(CORR_OUT, json.dumps(corrected_json, indent=2))
    session.add_event(sid, "Correction completed.")
    logger.info("Correction agent output saved.")
except Exception as e:
    logger.error(f"Correction agent failed: {e}")
    raise



# 5. EVALUATION (IoU)

# Note: Ensure 'from PIL import Image' is at the top of your main.py file.
# The IMAGE_PATH and GT_PATH variables are assumed to be defined earlier.

iou_score = None

try:
    if corrected_json.get("objects"):
        # 1. Open image to get dimensions for normalized -> pixel conversion
        img = Image.open(IMAGE_PATH)
        W, H = img.size
        
        with open(GT_PATH, "r") as f:
            gt = json.load(f)

        # 2. PRED: Extract normalized coordinates
        x_norm, y_norm, w_norm, h_norm = corrected_json["objects"][0]["bbox_norm"]
        
        # 3. PRED: Convert normalized prediction (x_norm, y_norm, w_norm, h_norm) to pixel (x, y, w, h)
        pred_bbox = [
            x_norm * W,
            y_norm * H,
            w_norm * W,
            h_norm * H
        ]

        # 4. GT: Assuming ground truth is in pixel format [x, y, w, h] as expected by eval.py:iou()
        # If your ground truth is also normalized, you would need to convert it here too.
        gt_bbox = gt["objects"][0]["bbox"] 

        iou_score = iou(pred_bbox, gt_bbox)
        session.add_event(sid, f"IoU: {iou_score}")
        remember("last_iou", iou_score)
        logger.info(f"IoU Score = {iou_score:.4f}")
    else:
        logger.warning("No predicted objects for IoU.")
except FileNotFoundError:
    logger.warning("GT file not found → IoU skipped.")
except Exception as e:
    logger.error(f"IoU computation error: {e}")



# 6. YOLO CONVERSION

try:
    yolo_txt = convert_to_yolo(json.dumps(corrected_json), IMAGE_PATH)
    save_text(YOLO_OUT, yolo_txt)
    session.add_event(sid, "YOLO conversion completed.")
    logger.info("YOLO output saved.")
except Exception as e:
    logger.error(f"YOLO conversion failed: {e}")



# 7. SAVE SESSION TO MEMORY BANK
remember("last_run", {
    "session_id": sid,
    "timestamp": datetime.now().isoformat(),
    "events": session.get_session(sid)["events"],
    "final_output": corrected_json
})

logger.info("Annotation pipeline completed successfully.")
print("Annotation complete! Session saved in memory bank under 'last_run'.")

