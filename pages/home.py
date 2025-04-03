import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Raga Classification System",
    page_icon="🎵",
    layout="wide"
)

# Header section
st.markdown("<h1 style='text-align: center;'>Raga Classification System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #636EFA;'>Deep Learning for Indian Classical Music Analysis</h3>", unsafe_allow_html=True)

# Documentation section
with st.expander("📚 Project Documentation", expanded=True):
    st.markdown("""
    ## Project Overview
    
    This system uses deep learning techniques to classify Indian ragas from audio recordings. The project pipeline includes:
    
    1. **Audio Preprocessing**: Converting MP3 files into 30-second segments and extracting chromagrams
    2. **Feature Engineering**: Transforming audio data into chromatic representations
    3. **Model Architecture**: Using a hybrid CNN-LSTM neural network for classification
    4. **Training & Evaluation**: Training on a dataset of ragas with validation and testing
    
    ### Key Features
    
    - **Multi-Raga Classification**: Supports identification of numerous raga types
    - **Segment-Based Analysis**: Processes audio in 30-second windows for consistent analysis
    - **Chromagram Feature Extraction**: Utilizes chromatic information crucial for raga identification
    - **Deep Learning Model**: Employs a sophisticated CNN-LSTM architecture
    
    ### Technical Documentation
    
    The preprocessing pipeline takes raw MP3 files, segments them into 30-second clips, and extracts chromagrams using Librosa's chroma_stft function. These features are then fed into a CNN-LSTM model for classification.
    
    ```python
    # Sample code for chromagram extraction
    chroma = librosa.feature.chroma_stft(y=segment_audio, sr=SAMPLE_RATE, n_chroma=12, n_fft=4096)
    ```
    
    The model architecture combines convolutional layers for feature extraction and LSTM layers for sequence modeling, making it well-suited for music pattern recognition.
    """)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Architecture", "Data Processing", "Implementation"])

with tab1:
    st.markdown("## Model Architecture")
    
    # Model architecture visualization
    architecture_data = pd.DataFrame({
        "Layer Type": ["Input", "Conv2D", "BatchNorm", "Conv2D", "BatchNorm", "MaxPooling2D", 
                      "Conv2D", "BatchNorm", "Conv2D", "BatchNorm", "MaxPooling2D", 
                      "Conv2D", "BatchNorm", "Flatten", "Reshape", "LSTM", "LSTM", "LSTM", 
                      "Dense", "Dropout", "Dense (Softmax)"],
        "Output Shape": ["(12, 1292, 1)", "(12, 1292, 32)", "(12, 1292, 32)", "(12, 1292, 32)", "(12, 1292, 32)",
                        "(6, 646, 32)", "(6, 646, 64)", "(6, 646, 64)", "(6, 646, 64)", "(6, 646, 64)",
                        "(3, 323, 64)", "(3, 323, 64)", "(3, 323, 64)", "(62016)", "(12, 5168)", 
                        "(12, 64)", "(12, 64)", "(32)", "(128)", "(128)", "(95)"],
        "Parameters": ["0", "320", "128", "9,248", "128", "0", "18,496", "256", "36,928", "256", 
                      "0", "36,928", "256", "0", "0", "1,327,104", "33,024", "12,416", "4,224", "0", "12,255"]
    })
    
    st.dataframe(architecture_data, use_container_width=True)
    
    # Model layers visualization
    st.markdown("### Model Layer Distribution")
    layers = ["Input", "Conv2D", "BatchNorm", "MaxPool", "LSTM", "Dense", "Output"]
    layer_counts = [1, 5, 5, 2, 3, 2, 1]
    
    fig = px.bar(
        x=layers, 
        y=layer_counts,
        labels={"x": "Layer Type", "y": "Count"},
        color=layers,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Architecture Overview:**
    - **Input**: Chromagram representation (12×1292×1)
    - **Feature Extraction**: Multiple Conv2D layers with BatchNorm
    - **Temporal Processing**: 3 LSTM layers for sequential pattern recognition
    - **Classification**: Fully connected layers with dropout for regularization
    - **Output**: 95 raga classes with softmax activation
    
    **Key Architectural Benefits:**
    - The convolutional layers extract local patterns from the chromagram
    - BatchNorm improves training stability and speed
    - LSTM layers capture the temporal progression of notes essential to raga identification
    - Dropout (0.5) prevents overfitting on the training dataset
    """)

with tab2:
    st.markdown("## Data Processing Pipeline")
    
    # Processing steps visualization
    steps = [
        "MP3 Loading",
        "30s Segmentation",
        "Chromagram Extraction",
        "Feature Normalization",
        "Data Augmentation",
        "Model Training"
    ]
    
    # Create a flowchart-like visualization
    fig = go.Figure(go.Scatter(
        x=[0, 1, 2, 3, 4, 5],
        y=[0, 0, 0, 0, 0, 0],
        mode="markers+lines+text",
        marker=dict(size=30, color=px.colors.qualitative.Safe),
        line=dict(width=4, color="gray"),
        text=steps,
        textposition="top center",
        textfont=dict(size=14)
    ))
    
    fig.update_layout(
        title="Audio Processing Pipeline",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Preprocessing steps
    st.markdown("""
    ### Preprocessing Steps
    
    1. **Audio Loading**
       - Uses librosa to load MP3 files at 22050Hz sample rate
       - Handles files of various durations and quality
       - Automatically resamples audio from different sources to a consistent rate
    
    2. **Segmentation**
       - Divides audio into 30-second segments
       - Ensures consistent input size for the model
       - Discards recordings shorter than the target duration
       - Uses overlapping segments (15-second overlap) to create more training data
    
    3. **Feature Extraction**
       - Computes chromagrams (12-dimensional representation of pitch content)
       - Uses n_fft=4096 for frequency resolution
       - Captures pitch class information crucial for raga identification
       - Normalizes features to enhance model learning
    
    4. **Data Augmentation**
       - Pitch shifting (±1 semitone)
       - Time stretching (±5%)
       - Adds slight noise for robustness
       - Creates 3x more training samples
    
    5. **Data Storage**
       - Segments are saved as MP3 files for future use
       - Chromagrams are stored in memory for immediate model training
       - Final processed data is cached to avoid redundant processing
    """)
    
    # Display sample statistics
    st.markdown("### Dataset Statistics")
    data_stats = pd.DataFrame({
        "Metric": ["Total Ragas", "Total Recordings", "Segments Created", "Feature Dimension", "Segment Duration", "Sample Rate", "Training Samples", "Validation Samples", "Test Samples"],
        "Value": ["95", "500+", "2000+", "(12, 1292)", "30 seconds", "22050 Hz", "1600", "200", "200"]
    })
    st.dataframe(data_stats, use_container_width=True)
    
    # Sample data visualization
    st.markdown("### Sample Chromagram")
    # Generate a sample chromagram for visualization
    y = np.sin(2 * np.pi * np.arange(0, 22050*5) * 440 / 22050)
    sample_chroma = librosa.feature.chroma_stft(y=y, sr=22050, n_chroma=12, n_fft=4096)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(sample_chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set_title('Chromagram Representation')
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

with tab3:
    st.markdown("## Implementation Details")
    
    st.markdown("""
    ### Technology Stack
    
    - **Programming Language**: Python 3.8+ (≤ 11)
    - **Audio Processing**: Librosa, SoundFile
    - **Data Handling**: NumPy, Pandas
    - **Model Development**: TensorFlow 2.x, Keras
    - **Visualization**: Matplotlib, Plotly
    - **Web Interface**: Streamlit
    - **Deployment**: Docker containerization for easy deployment
    - **Version Control**: Git with GitHub Actions for CI/CD
    """)
    
    # Technology usage visualization
    tech_data = pd.DataFrame({
        "Technology": ["Python", "TensorFlow/Keras", "Librosa", "NumPy/Pandas", "Streamlit", "Matplotlib/Plotly", "Docker", "Git/CI"],
        "Usage (%)": [20, 25, 20, 10, 10, 5, 5, 5]
    })
    
    fig = px.pie(
        tech_data, 
        values="Usage (%)", 
        names="Technology",
        title="Technology Distribution",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Code execution timeline
    st.markdown("### Execution Timeline")
    
    timeline_data = pd.DataFrame({
        "Stage": ["Data Loading", "Preprocessing", "Feature Extraction", "Model Training", "Evaluation", "Model Export"],
        "Time (mins)": [5, 20, 15, 60, 10, 2]
    })
    
    fig = px.bar(
        timeline_data,
        x="Stage",
        y="Time (mins)",
        title="Processing Time by Stage",
        color="Stage",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### System Requirements
    
    - **CPU**: 4+ cores recommended
    - **RAM**: 8GB+
    - **Storage**: 5GB+ for dataset and processed files
    - **GPU**: Optional but recommended for faster training
    - **Libraries**: TensorFlow, Librosa, NumPy, SoundFile
    
    ### Project Structure
    
    ```
    raga_classification/
    ├── data/                       # Not included in repository due to size
    │   ├── raw/                    # Raw MP3 files by raga
    │   │   ├── Bhairav/
    │   │   ├── Yaman/
    │   │   └── ...
    │   ├── processed/              # Segmented MP3 files
    │   └── features/               # Extracted features
    ├── models/
    │   ├── raga_model.h5           # Trained model
    │   └── checkpoints/            # Training checkpoints
    ├── accuracy_plot.png           # Training accuracy visualization
    ├── loss_plot.png               # Training loss visualization
    ├── dl_raga_classifier_core_collab.ipynb  # Jupyter notebook with core model
    ├── frontend.py                 # Streamlit UI implementation
    ├── main.py                     # Main program entry point
    ├── preprocess.py               # Dataset preprocessing script
    ├── requirements.txt            # Project dependencies
    ├── README.md                   # Project documentation
    └── LICENSE                     # License information
    ```
    
    ### Instructions
    
    1. **Setup Environment**:
       ```bash
       # Create virtual environment with Python ≤ 11
       python -m venv venv
       
       # Activate virtual environment
       # On Windows
       venv\\Scripts\\activate
       # On macOS/Linux
       source venv/bin/activate
       
       # Install dependencies
       pip install -r requirements.txt
       ```
    
    2. **Data Preparation**:
       ```bash
       # Place your raga audio files in a data folder structure:
       # data/raw/Bhairav/
       # data/raw/Yaman/
       # etc.
       
       # Run preprocessing to segment audio files
       python preprocess.py
       ```
    
    3. **Model Training**:
       ```bash
       # Train the raga classification model
       python main.py --mode train
       ```
    
    4. **Launch Dashboard**:
       ```bash
       # Start the Streamlit application
       streamlit run frontend.py
       ```
    
    5. **Inference**:
       ```bash
       # For inference on new audio files
       python main.py --mode predict --input path/to/audio.mp3
       ```
    
    ### Model Usage
    
    The trained model can be loaded and used for inference in your own Python code:
    
    ```python
    import tensorflow as tf
    import librosa
    import numpy as np
    
    # Load the model
    model = tf.keras.models.load_model('models/raga_model.h5')
    
    def predict_raga(audio_path):
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        
        # Reshape for model input
        chroma = np.expand_dims(chroma, axis=-1)
        
        # Predict
        prediction = model.predict(np.expand_dims(chroma, axis=0))
        raga_index = np.argmax(prediction[0])
        
        # Get raga name (replace with your raga mapping)
        ragas = ['Bhairav', 'Yaman', 'Bhairavi', '...']
        return ragas[raga_index]
    ```
    """)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>Raga Classification System | Developed with TensorFlow, Librosa, and Streamlit</p>
    <p>© 2025 Indian Classical Music Research Project</p>
</div>
""", unsafe_allow_html=True)