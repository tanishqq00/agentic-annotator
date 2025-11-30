# app.py — Streamlit UI v3 (Production Grade)

import os
import json
import logging
from datetime import datetime
from PIL import Image
import streamlit as st


# Logging 

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


from src.config import GOOGLE_API_KEY
import google.generativeai as genai

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing.")

genai.configure(api_key=GOOGLE_API_KEY)
logger.info("Gemini API key loaded ✔ (hidden)")


# Import Agents & Utilities

from agents.planner_agent import make_plan
from agents.perception_agent import annotate_image
from agents.correction_agent import correct_annotation

from src.yolo_formatter import convert_to_yolo
from src.eval import iou
from src.tools import save_text
from src.visualize import draw_boxes
from src.session_service import InMemorySessionService
from src.memory_bank import recall, remember


# Streamlit Setup

st.set_page_config(page_title="Agentic Auto Annotation", layout="wide")
st.title("🧠 Agentic Auto-Annotation")


# Sidebar

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Single Image", "Batch"])
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
run_iou = st.sidebar.checkbox("Compute IoU (gt_sample.json required)", True)
auto_save = st.sidebar.checkbox("Auto-Save Results", True)

st.sidebar.markdown("---")
st.sidebar.header("Memory")
if st.sidebar.button("Show last IoU"):
    st.sidebar.write(recall("last_iou"))

# Create Session

session = InMemorySessionService()
sid = session.create_session("streamlit_user")
session.add_event(sid, "UI session started")


# Layout Columns

left, right = st.columns((2, 1))

# LEFT COLUMN — Upload + Planner

with left:
    st.subheader("Image Input")

    if mode == "Single Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Sample selection
    st.markdown("**Or choose existing image in data/ui:**")
    os.makedirs("data/ui", exist_ok=True)
    ui_images = [f for f in os.listdir("data/ui") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = st.selectbox("Choose sample", ["-- none --"] + ui_images)

    run_btn = st.button("🚀 Run Annotation")

    st.markdown("---")
    st.subheader("Planner")
    try:
        plan_json = make_plan("data/ui/sample")
        st.json(json.loads(plan_json))
    except Exception as e:
        st.error("Planner failed.")
        logger.error(f"Planner failed: {e}")


# RIGHT COLUMN — Events + Outputs

with right:
    st.subheader("Session Events")
    session_box = st.empty()

    st.subheader("Last Output")
    last_output_box = st.empty()


# Helper: Process image

def process_image(image_path):
    session.add_event(sid, f"Processing {image_path}")

    
    # 1. Perception
   
    try:
        raw = annotate_image(image_path)
        session.add_event(sid, "Perception completed")
    except Exception as e:
        logger.error(f"Perception agent failed: {e}")
        return {"error": f"Perception failed: {e}"}

    
    # 2. Correction
    
    try:
        corrected = correct_annotation(image_path, raw)
    except Exception as e:
        logger.error(f"Correction agent failed: {e}")
        return {"error": f"Correction failed: {e}"}

    # Parse JSON safely
    try:
        corrected_json = json.loads(corrected)
    except Exception:
        import re
        match = re.search(r"(\{[\s\S]*\})", corrected)
        corrected_json = json.loads(match.group(0)) if match else {"objects": []}

    
    # 3. IoU Evaluation
    
    iou_score = None
    if run_iou and os.path.exists("annotations/gt_sample.json"):
        try:
            with open("annotations/gt_sample.json") as f:
                gt = json.load(f)
            pred = corrected_json["objects"][0]["bbox"]
            truth = gt["objects"][0]["bbox"]
            iou_score = iou(pred, truth)
            remember("last_iou", iou_score)
            session.add_event(sid, f"IoU: {iou_score}")
        except Exception as e:
            logger.error(f"IoU failed: {e}")

   
    # 4. YOLO Conversion
    
    try:
        yolo_txt = convert_to_yolo(json.dumps(corrected_json), image_path)
    except Exception as e:
        logger.error("YOLO conversion failed.")
        yolo_txt = ""

    
    # 5. Visualization
    
    boxed = None
    if show_boxes:
        try:
            boxed = draw_boxes(image_path, corrected_json)
        except Exception:
            logger.warning("Box drawing failed.")

    
    # 6. Save
   
    base = os.path.splitext(os.path.basename(image_path))[0]
    if auto_save:
        os.makedirs("annotations", exist_ok=True)
        save_text(f"annotations/{base}_raw.json", raw)
        save_text(f"annotations/{base}_corrected.json", json.dumps(corrected_json, indent=2))
        save_text(f"annotations/{base}.txt", yolo_txt)

    return {
        "raw": raw,
        "corrected": corrected_json,
        "iou": iou_score,
        "yolo": yolo_txt,
        "image": boxed
    }


# RUN PIPELINE

if run_btn:
    results = []
    
    if mode == "Single Image":
        if uploaded:
            img_path = f"data/ui/{uploaded.name}"
            with open(img_path, "wb") as f:
                f.write(uploaded.getbuffer())
            results.append((img_path, process_image(img_path)))

        elif sample != "-- none --":
            img_path = os.path.join("data/ui", sample)
            results.append((img_path, process_image(img_path)))

        else:
            st.error("Upload or select an image to run.")
    else:
        if uploaded:
            for f in uploaded:
                img_path = f"data/ui/{f.name}"
                with open(img_path, "wb") as out:
                    out.write(f.getbuffer())
                results.append((img_path, process_image(img_path)))
        elif sample != "-- none --":
            img_path = os.path.join("data/ui", sample)
            results.append((img_path, process_image(img_path)))
        else:
            st.error("Upload or select images for batch mode.")

    
    # DISPLAY RESULTS
    
    for img_path, res in results:
        with st.expander(f"Results — {os.path.basename(img_path)}", expanded=True):
            if "error" in res:
                st.error(res["error"])
                continue

            st.subheader("Corrected Annotation")
            st.json(res["corrected"])

            if res["image"] is not None:
                st.subheader("Annotated Image")
                st.image(res["image"], use_container_width=True)

            st.subheader("YOLO Output")
            st.code(res["yolo"])

            if res["iou"] is not None:
                st.success(f"IoU: {res['iou']}")

    session_box.text("\n".join([f"{e['ts']}: {e['event']}" 
                                for e in session.get_session(sid)["events"]]))

    st.success("Done ✔")

# VIEW SESSION LOGS

with st.expander("Session Log"):
    for e in session.get_session(sid)["events"]:
        st.write(f"- {e['ts']}: {e['event']}")


