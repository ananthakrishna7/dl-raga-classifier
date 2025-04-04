import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import pickle


MODEL_PATH = "./raga_model5.keras"
ENCODER_PATH = "./label_encoder.pkl"
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


PREPROCESSED_DIR = "./preprocessed"


RAGA_MAPPING_FILE = "raga_mapping.json"
if os.path.exists(RAGA_MAPPING_FILE):
    with open(RAGA_MAPPING_FILE, "r") as f:
        raga_mapping = json.load(f)
else:
    raga_mapping = {}

st.title("🎵 Upload Audio Clip to Find Raga")


uploaded_file = st.file_uploader("Select an Audio File:", type=["mp3"])

if uploaded_file:
    
    temp_file_path = os.path.join(PREPROCESSED_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    
    if st.button("Check"):
        try:
           
            y, sr = librosa.load(temp_file_path, sr=22050)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)  

            st.markdown("### 📊 Computed Chromagram")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", cmap="coolwarm")
            ax.set_title(f"Chromagram - {uploaded_file.name}")
            st.pyplot(fig)

            expected_length = 1292
            if chroma.shape[1] < expected_length:
                padding = expected_length - chroma.shape[1]
                chroma = np.pad(chroma, ((0, 0), (0, padding)))
            else:
                chroma = chroma[:, :expected_length]

            chroma_input = chroma[np.newaxis, :, :]  

            prediction = model.predict(chroma_input)
            predicted_index = np.argmax(prediction)
            predicted_raga = label_encoder.inverse_transform([predicted_index])[0]

            st.markdown(f"### 🎼 Predicted Raga: **{predicted_raga}**")

            original_chroma_path = os.path.join("preprocessed_chromograms/", uploaded_file.name.replace(".mp3", ".png"))
            if os.path.exists(original_chroma_path):
                st.markdown("### 📊 Original Preprocessed Chromogram")
                st.image(original_chroma_path, caption=f"Original Chromogram - {predicted_raga}", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

        finally:
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
