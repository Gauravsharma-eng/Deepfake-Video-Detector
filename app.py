# app.py

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
import yt_dlp
import requests
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# ---------------- MODEL ----------------

class DeepfakeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):

        return self.model(x)

    def load_model(self, path, device="cpu"):

        self.model.load_state_dict(
            torch.load(path, map_location=device)
        )

        self.model.to(device)

        self.model.eval()

    def predict(self, frame, device="cpu"):

        img = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (224, 224)
        )

        img = img / 255.0

        img = np.transpose(
            img,
            (2, 0, 1)
        )

        img = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            logits = self.forward(img)

            probs = F.softmax(
                logits,
                dim=1
            )

            conf, class_idx = torch.max(
                probs,
                dim=1
            )

            return class_idx.item(), conf.item()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    page_icon="🎭"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

body, .stApp {
    background-color: #121212;
    color: #f5f5f5;
}

.stButton>button {
    background-color: #ff0000;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff3333;
}

.big-title {
    font-size: 55px;
    font-weight: bold;
    color: #ffcc00;
    text-align: center;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MENU ----------------

selected = option_menu(
    menu_title=None,
    options=[
        "🏠 Home",
        "🕵️ Detection Tool",
        "ℹ️ Features"
    ],
    icons=[
        "house",
        "camera-video",
        "info-circle"
    ],
    orientation="horizontal",
)

# ---------------- HOME ----------------

if selected == "🏠 Home":

    st.markdown("""
    <div class='big-title'>
    🎬 Welcome to AI Deepfake Detector
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Upload video OR paste video link to detect deepfake videos."
    )

# ---------------- DETECTION TOOL ----------------

elif selected == "🕵️ Detection Tool":

    st.markdown("""
    <h2 style='text-align:center;color:#00ffcc;'>
    Upload OR Paste Video Link
    </h2>
    """, unsafe_allow_html=True)

    # ---------------- DEVICE ----------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    st.success(f"✅ Using Device: {device}")

    # ---------------- LOAD MODEL ----------------

    @st.cache_resource
    def load_model():

        detector = DeepfakeDetector()

        with st.spinner(
            "🔄 Loading AI Model..."
        ):

            time.sleep(2)

            detector.load_model(
                "checkpoints/model_best.pth",
                device=device
            )

        return detector

    model = load_model()

    # ---------------- VIDEO UPLOAD ----------------

    st.subheader("📤 Upload Video")

    uploaded_file = st.file_uploader(
        "Upload MP4 / AVI / MOV",
        type=["mp4", "avi", "mov"]
    )

    st.markdown("---")

    # ---------------- VIDEO URL ----------------

    st.subheader("🔗 Paste Video Link")

    video_url = st.text_input(
        "Paste YouTube or MP4 URL"
    )

    download_btn = st.button(
        "📥 Download & Analyze"
    )

    st.markdown("---")

    # ---------------- THRESHOLD ----------------

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.5
    )

    # ---------------- VIDEO PATH ----------------

    video_path = None

    # -------- URL VIDEO --------

    if download_btn and video_url:

        st.info("⏳ Downloading Video...")

        try:

            # YouTube

            if (
                "youtube.com" in video_url
                or
                "youtu.be" in video_url
            ):

                ydl_opts = {
                    'outtmpl': 'downloaded_video.mp4',
                    'format': 'mp4'
                }

                with yt_dlp.YoutubeDL(
                    ydl_opts
                ) as ydl:

                    ydl.download([video_url])

                video_path = "downloaded_video.mp4"

            # Direct MP4 Link

            else:

                response = requests.get(
                    video_url
                )

                with open(
                    "downloaded_video.mp4",
                    "wb"
                ) as f:

                    f.write(response.content)

                video_path = "downloaded_video.mp4"

            st.success(
                "✅ Video Downloaded"
            )

        except Exception as e:

            st.error(
                f"❌ Error: {e}"
            )

    # -------- UPLOAD VIDEO --------

    elif uploaded_file is not None:

        tfile = tempfile.NamedTemporaryFile(
            delete=False
        )

        tfile.write(
            uploaded_file.read()
        )

        video_path = tfile.name

    # ---------------- ANALYSIS ----------------

    if video_path is not None:

        st.video(video_path)

        output_dir = "outputs"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            st.error(
                "❌ Error opening video."
            )

            st.stop()

        fps = cap.get(
            cv2.CAP_PROP_FPS
        ) or 25

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count == 0:

            st.error(
                "❌ Video has 0 frames."
            )

            st.stop()

        frames = []

        labels = []

        confidences = []

        stframe = st.empty()

        progress_bar = st.progress(0)

        chart_placeholder = st.empty()

        # ---------------- FRAME LOOP ----------------

        for i in range(frame_count):

            ret, frame = cap.read()

            if not ret:
                break

            class_idx, conf = model.predict(
                frame,
                device=device
            )

            label = (
                "Real"
                if class_idx == 0
                else "Fake"
            )

            display_label = (
                label
                if conf >= confidence_threshold
                else "Uncertain"
            )

            labels.append(
                display_label
            )

            confidences.append(conf)

            # -------- DRAW LABEL --------

            frame_disp = frame.copy()

            color = (
                (0, 255, 0)
                if label == "Real"
                else (0, 0, 255)
            )

            cv2.putText(
                frame_disp,
                f"{display_label}: {conf*100:.2f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            stframe.image(
                cv2.cvtColor(
                    frame_disp,
                    cv2.COLOR_BGR2RGB
                )
            )

            frames.append(frame_disp)

            progress_bar.progress(
                (i + 1) / frame_count
            )

            # -------- LIVE PIE CHART --------

            if i % 5 == 0:

                real_count = labels.count(
                    "Real"
                )

                fake_count = labels.count(
                    "Fake"
                )

                uncertain_count = labels.count(
                    "Uncertain"
                )

                fig, ax = plt.subplots()

                ax.pie(
                    [
                        real_count,
                        fake_count,
                        uncertain_count
                    ],
                    labels=[
                        "Real",
                        "Fake",
                        "Uncertain"
                    ],
                    autopct='%1.1f%%'
                )

                ax.set_title(
                    "Frame Analysis"
                )

                chart_placeholder.pyplot(
                    fig
                )

        cap.release()

        # ---------------- FINAL RESULTS ----------------

        total = len(labels)

        real_count = labels.count(
            "Real"
        )

        fake_count = labels.count(
            "Fake"
        )

        uncertain_count = labels.count(
            "Uncertain"
        )

        avg_conf = (
            np.mean(confidences) * 100
        )

        st.markdown(
            "## ✅ Analysis Complete"
        )

        st.success(
            f"🎞️ Total Frames: {total}"
        )

        st.success(
            f"🟢 Real Frames: {real_count}"
        )

        st.error(
            f"🔴 Fake Frames: {fake_count}"
        )

        st.warning(
            f"🟡 Uncertain Frames: {uncertain_count}"
        )

        st.info(
            f"📊 Average Confidence: {avg_conf:.2f}%"
        )

        # ---------------- SAVE VIDEO ----------------

        output_path = os.path.join(
            output_dir,
            f"annotated_{int(time.time())}.mp4"
        )

        if frames:

            height, width, _ = frames[0].shape

            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            for f in frames:

                out.write(f)

            out.release()

            st.video(output_path)

            with open(
                output_path,
                "rb"
            ) as file:

                st.download_button(
                    label="📥 Download Annotated Video",
                    data=file,
                    file_name="deepfake_result.mp4",
                    mime="video/mp4"
                )

            st.success(
                f"✅ Annotated Video Saved"
            )

# ---------------- FEATURES ----------------

elif selected == "ℹ️ Features":

    st.write("""
    ✅ Upload Video Detection

    ✅ YouTube Video Detection

    ✅ MP4 Link Detection

    ✅ Frame-wise AI Detection

    ✅ Live Pie Chart

    ✅ Annotated Video Output

    ✅ Download Result Video

    ✅ GPU Support

    ✅ Dark Professional UI
    """)

    st.info(
        "Made by Gaurav Sharma 🚀"
    )
