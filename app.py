import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
import tempfile
import os
import time
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# --- Model Definition ---
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

    def load_model(self, path, device="cpu"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, frame, device="cpu"):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(img)
            probs = F.softmax(logits, dim=1)
            conf, class_idx = torch.max(probs, dim=1)
            return class_idx.item(), conf.item()

# --- Custom Styling ---
st.markdown("""
<style>
body, .stApp { background-color: #121212; color: #f5f5f5; }
.stButton>button { background-color: #ff0000; color: #fff; border-radius: 10px; border: none; padding: 8px 20px; font-weight: bold;}
.big-title { font-size: 55px; font-weight: bold; color: #ffcc00; text-shadow: 0 0 30px #ffcc00, 0 0 60px #ff0000; text-align: center; margin-bottom: 20px; }
.blink-msg { font-size: 22px; color: #ffcccc; text-align: center; animation: pulse 2s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "🕵️ Detection Tool", "ℹ️ Features"],
    icons=["house", "camera-video", "info-circle"],
    orientation="horizontal",
)

if selected == "🏠 Home":
    st.markdown("""
    <style>
    .eyes-container { display: flex; justify-content: center; margin: 40px 0; }
    .eye {
        width: 120px; height: 120px;
        background: radial-gradient(circle at center, #ff0000 25%, #220000 100%);
        border-radius: 50%; margin: 0 30px; position: relative; overflow: hidden;
        box-shadow: 0 0 60px #ff0000, inset 0 0 40px #ff0000;
        animation: eyePulse 3s infinite alternate;
    }
    @keyframes eyePulse { 0% { transform: scale(1); box-shadow: 0 0 40px #ff0000; } 100% { transform: scale(1.1); box-shadow: 0 0 100px #ff3333; } }
    .pupil { width: 40px; height: 40px; background: radial-gradient(circle at center, #000 30%, #660000 100%);
        border-radius: 50%; position: absolute; top: 40px; left: 40px; animation: pupilMove 6s infinite; }
    @keyframes pupilMove { 0%,100% { transform: translate(0,0) scale(1); } 25% { transform: translate(15px,-10px) scale(1.2); } 50% { transform: translate(-20px,15px) scale(0.9); } 75% { transform: translate(10px,20px) scale(1.3); } }
    .eyelid { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #000; border-radius: 50%; opacity: 0.9; animation: blink 4s infinite; }
    @keyframes blink { 0%, 90%, 100% { height: 0; } 45%, 50% { height: 100%; } }
    </style>
    <div class='eyes-container'>
        <div class='eye'><div class='pupil'></div><div class='eyelid'></div></div>
        <div class='eye'><div class='pupil'></div><div class='eyelid'></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='big-title'>🎬 Welcome to Our Deepfake Video Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='blink-msg'>Click on <b>Detection Tool</b> to continue 🚨</div>", unsafe_allow_html=True)

elif selected == "🕵️ Detection Tool":
    st.markdown("<h2 style='text-align:center;color:#00ffcc;'>Upload & Analyze Your Video</h2>", unsafe_allow_html=True)

    # Horror Eyes Effect
    st.markdown("""
    <style>
    .evil-eyes-container { display: flex; justify-content: center; margin: 30px 0; }
    .evil-eye { width: 100px; height: 100px; background: radial-gradient(circle at center, #ff0000 25%, #220000 100%);
        border-radius: 50%; margin: 0 40px; position: relative; overflow: hidden;
        box-shadow: 0 0 50px #ff0000, inset 0 0 30px #ff0000; animation: evilPulse 3s infinite alternate; }
    @keyframes evilPulse { 0% { transform: scale(1); box-shadow: 0 0 30px #ff0000; } 100% { transform: scale(1.1); box-shadow: 0 0 80px #ff3333; } }
    .evil-pupil { width: 30px; height: 30px; background: radial-gradient(circle at center, #000 30%, #660000 100%);
        border-radius: 50%; position: absolute; top: 35px; left: 35px; animation: evilPupilMove 6s infinite; }
    @keyframes evilPupilMove { 0%,100% { transform: translate(0,0) scale(1); } 25% { transform: translate(10px,-10px) scale(1.2); } 50% { transform: translate(-15px,10px) scale(0.9); } 75% { transform: translate(5px,15px) scale(1.3); } }
    .evil-eyelid { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #000; border-radius: 50%; opacity: 0.9; animation: evilBlink 5s infinite; }
    @keyframes evilBlink { 0%, 90%, 100% { height: 0; } 45%, 50% { height: 100%; } }
    </style>
    <div class='evil-eyes-container'>
        <div class='evil-eye'><div class='evil-pupil'></div><div class='evil-eyelid'></div></div>
        <div class='evil-eye'><div class='evil-pupil'></div><div class='evil-eyelid'></div></div>
    </div>
    """, unsafe_allow_html=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"🔹 Using device: {device}")

    @st.cache_resource
    def load_model():
        detector = DeepfakeDetector()
        with st.spinner("🔄 Loading AI Model... Please wait..."):
            time.sleep(2)
            # Ensure this path exists in your repo
            detector.load_model("checkpoints/model_best.pth", device=device)
        return detector

    # FIXED INDENTATION: model is now defined inside the elif block
    model = load_model()

    # --- VIDEO INPUT SECTION ---
    input_option = st.radio("Choose Video Input Method", ["Upload Video", "Paste Video URL"])
    video_path = None

    if input_option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
    else:
        video_url = st.text_input("Paste Video URL")
        if video_url:
            video_path = video_url

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

    if video_path:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Error opening video file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames, labels, confidences = [], [], []
            stframe = st.empty()
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret: break

                class_idx, conf = model.predict(frame, device=device)
                label = "Real" if class_idx == 0 else "Fake"
                display_label = label if conf >= confidence_threshold else "Uncertain"

                labels.append(label)
                confidences.append(conf)

                frame_disp = frame.copy()
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)
                cv2.putText(frame_disp, f"{display_label}: {conf*100:.2f}%", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                stframe.image(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
                frames.append(frame_disp)
                progress_bar.progress((i+1)/frame_count)

                # Update Chart
                if i % 10 == 0 or i == frame_count - 1:
                    real_c = labels.count("Real")
                    fake_c = labels.count("Fake")
                    fig, ax = plt.subplots()
                    ax.pie([real_c, fake_c], labels=["Real", "Fake"], colors=["#00ff00", "#ff0000"], autopct='%1.1f%%')
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)

            cap.release()
            st.success("✅ Analysis Complete")

elif selected == "ℹ️ Features":
    st.markdown("### 🚀 Project Features")
    st.write("""
    - **Real-time Detection:** Frame-by-frame analysis using ResNet50.
    - **Visual Insights:** Live distribution charts and annotated video playback.
    - **Flexible Input:** Supports local uploads and URL processing.
    - **Hardware Optimized:** Automatically uses GPU (CUDA) if available.
    """)
    st.info("Developed by Gaurav Sharma & Team")
