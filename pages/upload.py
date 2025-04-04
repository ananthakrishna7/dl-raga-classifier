import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import json

# Path to preprocessed audio files
PREPROCESSED_DIR = "./preprocessed"

# Load Raga mappings (Assuming a JSON file with mappings exists)
RAGA_MAPPING_FILE = "raga_mapping.json"
if os.path.exists(RAGA_MAPPING_FILE):
    with open(RAGA_MAPPING_FILE, "r") as f:
        raga_mapping = json.load(f)
else:
    raga_mapping = {}  # Fallback in case the file doesn't exist

st.title("🎵 Upload Audio Clip to Find Raga")

# File uploader for selecting an audio file from preprocessed folder
uploaded_file = st.file_uploader("Select a Audio File:", type=["mp3"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(PREPROCESSED_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Check Button
    if st.button("Check"):
        try:
            # Load the selected audio file
            y, sr = librosa.load(temp_file_path, sr=22050)

            # Compute the Chromogram
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)

            # Display the Chromogram
            st.markdown("### 📊 Computed Chromogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", cmap="coolwarm")
            ax.set_title(f"Chromogram - {uploaded_file.name}")
            st.pyplot(fig)

            # Get the corresponding Raga name
            raga_name = raga_mapping.get(uploaded_file.name, "Unknown Raga")
            st.markdown(f"### 🎼 Detected Raga: **{raga_name}**")

            # Display the preprocessed Chromogram (if stored separately)
            original_chroma_path = os.path.join("preprocessed_chromograms/", uploaded_file.name.replace(".mp3", ".png"))
            if os.path.exists(original_chroma_path):
                st.markdown("### 📊 Original Preprocessed Chromogram")
                st.image(original_chroma_path, caption=f"Original Chromogram - {raga_name}", use_column_width=True)

        finally:
            # Delete the file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
