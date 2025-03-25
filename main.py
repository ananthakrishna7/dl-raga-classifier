import librosa as lr
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

'''Assumes music files are under respective
raga folders, loads enough snippets of each 
raga to make all classes have equal training
samples in both train and test sets'''
# BUG: Reading from mp4
def load_preprocess(path: str, samples: int=10, train:float=0.8): # full path to ragas folder
    os.chdir(path)
    X_train, X_test = [], [] # chromagrams
    Y_train, Y_test = [], [] # ragas
    train_samples = int(samples*train)
    test_samples = samples - train_samples
    for folder in os.listdir():
        os.chdir(path + "/" + folder)
        songs = os.listdir()
        if (len(songs) < samples):
            snippets = []
            i = 0
            offset = 0
            length = 0
            while (length < samples): # adjust this in case of librosa errors, there may not be songs long enough. can also reduce snippet size
                print(i)
                mus, sr = lr.load(songs[i], duration=30.0, offset=30.0*i)
                chromagram = lr.feature.chroma_cqt(y=mus, sr=sr)
                snippets.append(chromagram)
                length += 1
                i += 1
                if i >= len(songs) - 1:
                    i = 0
                    offset += 1
            # we will have required no. of songs now (hopefully)
            X_train.extend(snippets[:train_samples])
            X_test.extend(snippets[train_samples:])
            Y_train.extend([folder for _ in range(train_samples)]) # that many songs from the same raga
            Y_test.extend([folder for _ in range(test_samples)]) # that many songs from the same raga
        else: # load all songs; the more the merrier ;)
            snippets = []
            for song in songs:
                mus, sr = lr.load(song, duration=30.0)
                chromagram = lr.feature.chroma_cqt(y=mus, sr=sr)
                snippets.append(chromagram)
            X_train.extend(snippets[:train_samples])
            X_test.extend(snippets[train_samples:])
            Y_train.extend([folder for _ in range(train_samples)]) # that many songs from the same raga
            Y_test.extend([folder for _ in range(test_samples)]) # that many songs from the same raga
    return X_train, X_test, Y_train, Y_test # change to np.array if required

if __name__ == "__main__":
    Xtr, Xts, Ytr, Yts = load_preprocess("/home/ananthakrishnan/sem6/Neural Networks and Deep Learning/project/ragas")
    print(Xtr, Ytr)