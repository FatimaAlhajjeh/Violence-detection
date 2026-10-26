import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Parameters
img_height, img_width = 224, 224  # Resize dimensions
frame_count = 25  # Number of frames to extract

# Helper function to preprocess video with uniform frame extraction
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames in the video
    frames = []

    if total_frames >= frame_count:
        # Uniformly sample 25 frames
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set frame position
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = cv2.resize(frame, (img_width, img_height))  # Resize frame
                frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize pixel values
                frames.append(frame)
    else:
        # Extract all frames, then pad with zeros if there are fewer than 25 frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frame = cv2.resize(frame, (img_width, img_height))  # Resize frame
            frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize pixel values
            frames.append(frame)

        # Zero-pad if the video has fewer frames than required
        while len(frames) < frame_count:
            frames.append(np.zeros((img_height, img_width), dtype=np.float32))  # Use float32 for consistency

    cap.release()

    # Convert to NumPy array and add channel dimension
    return np.expand_dims(np.array(frames), axis=-1)  # Shape: (25, 224, 224, 1)

# Load the trained 3D-CNN model
@st.cache_resource
def load_3d_cnn_model():
    model_path = "model.h5"  # Replace with your model path
    return load_model(model_path)

# Streamlit app
st.title("Video Classification App")

# Video uploader in the main app area
uploaded_file = st.file_uploader("Choose a video file to classify", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.write("Video uploaded successfully!")
    st.video(uploaded_file)  # Display the uploaded video

    # Save the uploaded video temporarily
    video_temp_path = "temp_video.mp4"
    with open(video_temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess the video
    st.write("Preprocessing the video...")
    try:
        preprocessed_video = preprocess_video(video_temp_path)
        st.write("Video preprocessing completed successfully!")

        # Load the model
        st.write("Loading the model...")
        model = load_3d_cnn_model()

        # Make a prediction
        st.write("Predicting the label...")
        preprocessed_video = np.expand_dims(preprocessed_video, axis=0)  # Add batch dimension
        prediction = model.predict(preprocessed_video)
        print("============================================", prediction)
        # Interpret prediction
        confidence_score = prediction[0][0]
        predicted_label = "Class 1 (Violent)" if confidence_score > 0.5 else "Class 0 (Non-Violent)"

        st.write(f"Predicted Label: {predicted_label}")
        st.write(f"Confidence Score: {confidence_score:.2f}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
    
    # Clean up the temporary video file
    os.remove(video_temp_path)
else:
    st.write("Please upload a video file to get started.")
