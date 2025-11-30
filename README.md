<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Gemini%20Vision-API-success?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Multi--Agent-System-purple?style=for-the-badge" />
  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Docker-Containerized-informational?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>


# 🧠 Agentic Auto-Annotation System

The Agentic Auto-Annotation System is a multi-agent AI pipeline that automatically annotates images using Gemini Vision, refines the bounding boxes, evaluates annotation quality through IoU, converts results into YOLO format, and presents everything through a clean Streamlit UI.

It demonstrates real-world agent collaboration, observability, memory, context engineering, and secure API usage — suitable for hackathons, workshops, and production scenarios.

# 📌 Overview

This project automates the image-annotation workflow using an LLM-powered multi-agent pipeline. An uploaded image goes through:

A Planner Agent that generates the workflow steps

A Perception Agent using Gemini Vision to extract objects and bounding boxes

A Correction Agent that normalizes bounding boxes

An Evaluation module that computes IoU against ground truth

A YOLO formatter that converts the output for object detection training

A Streamlit UI for visualization and downloads

Visualization is done locally using Pillow, allowing users to preview bounding boxes in real time without sending generated images back to Gemini.

The system also includes session tracking, memory persistence, observability, and secure secret management using Google Cloud Secret Manager.

# 🌐 High-Level Architecture

                             ┌──────────────────────────┐
                             │       Streamlit UI       │
                             │  (Frontend + User Input) │
                             └───────────┬──────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │        Agentic Backend Engine          │
                     │   (Python Multi-Agent Architecture)    │
                     └───────────┬─────────────┬─────────────┘
                                 │             │
                                 ▼             ▼
         ┌─────────────────────────────┐   ┌──────────────────────────────┐
         │        Planner Agent        │   │      Perception Agent        │
         │ Generates execution steps   │   │   Gemini Vision annotation   │
         └───────────────┬────────────┘   └───────────────┬──────────────┘
                         │                                 │
                         ▼                                 ▼
           ┌──────────────────────────┐       ┌─────────────────────────────┐
           │     Correction Agent     │       │      Gemini Vision API      │
           │ Normalizes bounding box  │ <---->│  Image understanding engine │
           └───────────┬─────────────┘       └─────────────────────────────┘
                       │
                       ▼
       ┌────────────────────────────────────────────┐
       │          Evaluation Module (IoU)           │
       │ Compares prediction to GT annotations      │
       └───────────────────┬────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────────┐
        │         YOLO Formatter (Normalized)          │
        │ Converts output to YOLO training format      │
        └────────────────────┬─────────────────────────┘
                              │
                              ▼
          ┌────────────────────────────────────────────┐
          │      Visualization (PIL Bounding Boxes)    │
          │ Produces annotated image for UI preview    │
          └───────────────────┬────────────────────────┘
                              │
                              ▼
       ┌────────────────────────────────────────────────────┐
       │        Session Logger + Memory Bank                │
       │ Stores events, IoU scores, and agent outputs       │
       └────────────────────────────────────────────────────┘


                              ▼
         ┌────────────────────────────────────────────────────┐
         │               Docker Container Image               │
         │ Production-ready image for serverless hosting      │
         └───────────────────────┬────────────────────────────┘
                                 │
                                 ▼
       ┌──────────────────────────────────────────────────────────┐
       │         Google Cloud Run (Serverless Deployment)         │
       │ Auto-scaling, stateless execution, secure secrets        │
       │ via Google Secret Manager                                │
       └──────────────────────────────────────────────────────────┘



# 🧰 Tech Stack

## Core Technologies
• Python 3.11
• Gemini Vision API (Google Generative AI)
• Streamlit for UI
• Pillow (PIL) for image visualization

## Agentic Components
• Multi-agent pipeline (planner, perception, correction)
• IoU-based evaluation module
• Session system for observability
• Memory bank for persistence
• Context-controlled annotation prompts

## Deployment & Security
• Docker for containerization
• Google Cloud Run for serverless hosting
• Google Secret Manager for API key security
• .dockerignore & .gitignore optimized for safety

## Data Formats
• JSON for annotations
• YOLO normalized format for training datasets

# 🚀 Deployment Strategy

This project supports two production deployment paths.

## Cloud Run Deployment 

Build the Docker image locally

Push to Docker Hub or Artifact Registry

Store your API key in Google Secret Manager

Deploy using Cloud Run with secret injection

Example deploy command:

gcloud run deploy agentic-annotator \
  --image docker.io/<username>/agentic-annotator:latest \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY_SECRET:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated


## Features:
• Fully serverless
• Automatic scaling
• Secure secrets
• Fast global access

## Local Deployment

Run:

streamlit run app.py


The UI opens locally at:

http://localhost:8501

# 📂 Project Structure
agentic_annotator/
│
├── app.py                    # Streamlit UI  
├── main.py                   # Multi-agent pipeline script  
├── Dockerfile                # Production container  
├── requirements.txt          # Python dependencies  
├── .gitignore                # Clean repo  
├── .dockerignore             # Clean Docker builds  
│
├── agents/                   # Planner, Perception, Correction agents  
├── src/                      # Formatters, tools, eval, session, memory  
├── data/                     # Sample images  
└── annotations/             # Auto-generated outputs  

# ⚡ Features at a Glance

• Multi-agent system using LLMs
• Zero-shot image annotation with Gemini Vision
• Correction and normalization of bounding boxes
• Visualization of annotations on images
• YOLO format export for model training
• IoU evaluation against ground-truth
• Persistent memory and session logs
• Secure API key handling
• Dockerized and Cloud Run-ready

# 👤 Author

Developed by Tanishq


# 📄 License

This project is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for details.
