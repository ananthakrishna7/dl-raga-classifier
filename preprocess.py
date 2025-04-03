import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Paths
INPUT_FOLDER = "ragas"           
OUTPUT_FOLDER = "preprocessed"   
TARGET_DURATION = 30  # 30 seconds in seconds
SAMPLE_RATE = 22050              

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_raga_folder(raga_name, raga_path):
    """Processes all MP3 files in a given raga folder."""
    output_raga_path = os.path.join(OUTPUT_FOLDER, raga_name)
    os.makedirs(output_raga_path, exist_ok=True)
    
    mp3_files = [f for f in os.listdir(raga_path) if f.endswith(".mp3")]

    chromagram_data = []  # Store chromagrams for training
    
    with tqdm(mp3_files, desc=f"Processing {raga_name}") as progress_bar:
        for mp3_file in progress_bar:
            mp3_path = os.path.join(raga_path, mp3_file)

            try:
                # Load audio using librosa
                y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE)

                # Skip if audio is shorter than 30 seconds
                total_duration = librosa.get_duration(y=y, sr=sr)
                if total_duration < TARGET_DURATION:
                    continue

                # Split into multiple 30-sec segments
                num_segments = int(total_duration // TARGET_DURATION)

                for segment_idx in range(num_segments):
                    start_sample = segment_idx * TARGET_DURATION * sr
                    end_sample = start_sample + TARGET_DURATION * sr
                    segment_audio = y[int(start_sample):int(end_sample)]

                    # Save segment as MP3
                    segment_filename = f"{os.path.splitext(mp3_file)[0]}_part{segment_idx+1}.mp3"
                    segment_path = os.path.join(output_raga_path, segment_filename)
                    sf.write(segment_path, segment_audio, sr, format="MP3")

                    # Extract chromagram (keep in memory)
                    chroma = librosa.feature.chroma_stft(y=segment_audio, sr=sr, n_chroma=12, n_fft=4096)
                    chromagram_data.append((chroma, raga_name))  # Store for training

            except Exception as e:
                print(f"Error processing {mp3_file}: {e}")

    return chromagram_data

# Iterate through raga subfolders
all_chromagrams = []
raga_folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
for raga_folder in raga_folders:
    chroma_data = process_raga_folder(raga_folder, os.path.join(INPUT_FOLDER, raga_folder))
    all_chromagrams.extend(chroma_data)

print(f"Preprocessing complete! Processed MP3s saved in '{OUTPUT_FOLDER}'.")
