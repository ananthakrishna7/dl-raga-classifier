import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objects as go

# Set page title
st.set_page_config(page_title="Raga Classification", layout="wide")

# Main Title
st.markdown("<h2 style='text-align: center;'>🎶🎭🪘</h2>", unsafe_allow_html=True)
st.markdown("""
    <h2 style='text-align: center;'>Raga Classification using CNN + LSTM Architecture</h2>
    <h4 style='text-align: center; color: gray;'>A Deep Learning Approach for Indian Classical Music</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# Section: Model Architecture
col1, col2 = st.columns([1, 1])
with col1:
    model_data = pd.DataFrame({
        "Layer": ["Conv2D", "MaxPool", "LSTM", "Dense", "Softmax"],
        "Output Shape": ["(64, 64, 32)", "(32, 32, 32)", "(128)", "(64)", "(10)"]
    })
    st.dataframe(model_data)
with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Conv2D", "LSTM", "Dense", "Softmax"], y=[5000, 2000, 1000, 500], name="Params"))
    fig.update_layout(title="Model Layer Parameters", xaxis_title="Layer", yaxis_title="Parameter Count")
    st.plotly_chart(fig)

st.markdown("---")

# Section: Preprocessing Steps
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
    ## Preprocessing Steps
    - Audio signals are converted into **Mel Spectrograms** for feature extraction.
    - Noise reduction and normalization are applied to improve quality.
    - Data augmentation techniques like pitch shifting and time stretching are used.
    - Features are reshaped into tensors suitable for CNN-LSTM training.
    """)
with col2:
    # Generate dummy waveform and spectrogram
    y = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Dummy Waveform")
    st.pyplot(fig)

st.markdown("---")

# Section: Technologies Used
col1, col2 = st.columns([1, 1])
with col1:
    tech_data = pd.DataFrame({
        "Technology": ["Python", "TensorFlow", "Librosa", "Streamlit", "Matplotlib"],
        "Usage (%)": [30, 25, 20, 15, 10]
    })
    st.bar_chart(tech_data.set_index("Technology"))
with col2:
    st.markdown("""
    ## Technologies Used
    - **Python, Streamlit** for web application
    - **Librosa** for audio processing
    - **TensorFlow/Keras** for deep learning model
    - **Matplotlib, Seaborn** for data visualization
    - **Jupyter Notebook** for model experimentation
    """)
