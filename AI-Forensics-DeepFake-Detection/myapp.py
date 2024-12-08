import streamlit as st
import torch
from torch.utils.model_zoo import load_url
import numpy as np
from scipy.special import expit
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary modules (assumes same directory structure)
from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

def initialize_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    face_policy = 'scale'
    face_size = 224
    frames_per_video = 64

    model_url = weights.weight_url['EfficientNetAutoAttB4_DFDC']
    net = getattr(fornet, 'EfficientNetAutoAttB4')().eval().to(device)
    net.load_state_dict(torch.hub.load_state_dict_from_url(model_url, map_location=device, check_hash=True))

    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")

    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

    return net, transf, face_extractor, device

def process_video(video_path, net, transf, face_extractor, device):
    frames_per_video = 64
    vid_faces = face_extractor.process_video(video_path)

    faces_t = torch.stack([
        transf(image=frame['faces'][0])['image']
        for frame in vid_faces if len(frame['faces'])
    ])

    with torch.no_grad():
        faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()

    score = expit(faces_pred.mean())
    return score, faces_pred

def main():
    # Page configuration
    st.set_page_config(page_title="Deepfake Detector", layout="wide", page_icon=":video_camera:")
    
    # Sidebar
    st.sidebar.title("Deepfake Detector")
    st.sidebar.info("Upload a video to analyze its authenticity.")
    st.sidebar.markdown("""
        ### Instructions:
        - Upload an MP4, AVI, or MOV video.
        - Wait for processing to complete.
        - View results and per-frame analysis.
    """)

    st.title("🎥 Deepfake Detector")
    st.markdown("Identify **deepfake videos** with precision and speed. Upload a video to get started.")

    # Initialize model
    net, transf, face_extractor, device = initialize_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        video_path = os.path.join("temp_video", uploaded_file.name)
        os.makedirs("temp_video", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        st.markdown("### 📊 Processing Results")
        with st.spinner("Processing video..."):
            # Progress feedback
            progress_bar = st.progress(0)
            score, faces_pred = process_video(video_path, net, transf, face_extractor, device)
            progress_bar.progress(100)

        # Display results
        prediction = "FAKE" if score > 0.5 else "REAL"
        st.success(f"**Overall Score:** {score:.4f} - **Prediction:** {prediction}")

        # Per-frame analysis
        st.markdown("### Per-frame Analysis")
        frame_scores = expit(faces_pred)
        df = pd.DataFrame({"Frame": range(1, len(frame_scores) + 1), "Score": frame_scores})
        st.dataframe(df.style.format({"Score": "{:.4f}"}), use_container_width=True)

        # Plot per-frame scores
        st.markdown("#### Frame Score Chart")
        plt.figure(figsize=(10, 6))
        plt.plot(df["Frame"], df["Score"], marker='o', linestyle='-', color='b')
        plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
        plt.title("Per-frame Analysis")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
