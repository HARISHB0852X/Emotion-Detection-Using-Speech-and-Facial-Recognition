# Emotion-Detection-Using-Speech-and-Facial-Recognition
# Emotion Detection Using Speech and Facial Recognition

## Overview

This project is a multimodal emotion detection system that uses both facial images and speech audio to identify human emotions. It combines state-of-the-art deep learning models for facial expression recognition and large language models (LLMs) for speech transcription and emotion analysis. The system is designed for robust, real-world applications and features an interactive dashboard for easy user interaction.

---

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Facial Emotion Recognition](#facial-emotion-recognition)
4. [Speech Emotion Recognition](#speech-emotion-recognition)
5. [Pipeline](#pipeline)
6. [Dashboard / User Interface](#dashboard--user-interface)
7. [Dependencies & Installation](#dependencies--installation)
8. [Usage](#usage)
9. [Limitations](#limitations)
10. [Future Enhancements](#future-enhancements)
11. [Conclusion](#conclusion)

---

## Features

- **Multimodal Emotion Detection:** Classifies emotions using both facial images and speech audio.
- **Facial Emotion Recognition:** Uses a pre-trained Mini-XCEPTION CNN model for fast, accurate facial emotion classification.
- **Speech Emotion Recognition:** Leverages OpenAI’s Whisper LLM for speech-to-text and language detection, and SVM for emotion classification from audio features.
- **Multilingual Support:** Handles speech in multiple languages.
- **Interactive Dashboard:** Streamlit-based UI for uploading files and displaying results.
- **Visual Feedback:** Annotated images and clear text outputs.

---

## System Architecture

### High-Level Flow

```
[Input: Image or Audio]
        |
        v
[Preprocessing]
        |
        v
[Model Prediction (CNN or Whisper+SVM)]
        |
        v
[Emotion Output Display]
```

### Modules

- **Data Collection:** Uses FER2013 for facial images; synthetic data for audio demo.
- **Preprocessing:** Grayscale conversion, resizing for images; MFCC/pitch extraction for audio.
- **Model Inference:** Mini-XCEPTION CNN for images; Whisper + SVM for audio.
- **Dashboard:** Streamlit for user interaction and results visualization.

---

## Facial Emotion Recognition

- **Model:** Mini-XCEPTION CNN (pre-trained on FER2013)
- **Emotions Detected:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Pipeline:**
  1. Face detection with OpenCV
  2. Image preprocessing (grayscale, resize to 64x64, normalization)
  3. Emotion classification via CNN
  4. Annotated image output

**Sample Code:**
```python
import cv2
import numpy as np
from keras.models import load_model

model = load_model("emotion_model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load and preprocess image, detect faces, classify, and annotate
# (See detailed code in the project notebook)
```

---

## Speech Emotion Recognition

- **Transcription:** OpenAI Whisper LLM for accurate, multilingual speech-to-text and language detection.
- **Feature Extraction:** MFCC and pitch features via librosa.
- **Emotion Classification:** SVM classifier (demo uses synthetic data; replace with real labeled data for production).
- **Pipeline:**
  1. Convert audio to mono WAV, 16kHz
  2. Transcribe and detect language with Whisper
  3. Extract MFCC and pitch features
  4. Classify emotion with SVM
  5. Display transcription, language, and emotion

**Sample Code:**
```python
import whisper, librosa, numpy as np
from sklearn.svm import SVC
# See detailed code in the project notebook
```

---

## Pipeline

1. **Input:** User uploads an image (for facial emotion) or audio file (for speech emotion).
2. **Preprocessing:** Image is converted to grayscale and resized; audio is converted to mono WAV and features extracted.
3. **Prediction:** 
   - Image: Mini-XCEPTION CNN predicts emotion.
   - Audio: Whisper transcribes and detects language; SVM predicts emotion.
4. **Output:** Results displayed on dashboard with annotated images and text.

---

## Dashboard / User Interface

- **Built with Streamlit**
- Upload images or audio files
- See:
  - Detected emotion
  - Speech transcription and language (for audio)
  - Annotated image (for facial emotion)
- Simple, intuitive, and interactive

---

## Dependencies & Installation

**Required Libraries:**
- `keras`
- `opencv-python`
- `numpy`
- `streamlit`
- `openai-whisper`
- `librosa==0.10.0.post2`
- `scikit-learn`
- `pydub`
- `ffmpeg-python`

**Installation (Colab/Terminal):**
```bash
pip install keras opencv-python numpy streamlit openai-whisper librosa==0.10.0.post2 scikit-learn pydub ffmpeg-python
```

**Download Pre-trained Model:**
```bash
wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5
```

---

## Usage

1. **Facial Emotion Recognition:**
   - Upload an image via the dashboard.
   - The system detects faces, classifies emotions, and displays the annotated image.

2. **Speech Emotion Recognition:**
   - Upload an audio file (MP3/WAV/MP4).
   - The system transcribes the speech, detects the language, extracts features, and predicts the emotion.

3. **Run on Colab or Locally:**
   - Use the provided notebook or Streamlit app.
   - See code snippets in the project files for details.

---

## Limitations

- **Facial Model:**
  - Relies on frontal face detection.
  - Sensitive to lighting, occlusion, and pose.
  - May misclassify subtle or mixed emotions.
- **Speech Model:**
  - Demo SVM uses synthetic labels; real-world performance depends on labeled data.
  - Sensitive to audio quality and background noise.
- **General:**
  - No context-aware emotion interpretation.
  - Cannot handle sarcasm or complex emotional states.

---

## Future Enhancements

- Integrate real audio emotion datasets for robust SVM training.
- Replace CNN with Vision Transformers (ViT) for facial emotion.
- Add real-time video and audio stream support.
- Improve dataset diversity to reduce bias.
- Multimodal fusion (combine image, audio, and text for context-aware emotion detection).
- Temporal emotion tracking over sequences.

---

## Conclusion

This project demonstrates effective, multimodal emotion detection using modern deep learning and LLM-based speech recognition. While current limitations exist, the system provides a strong foundation for emotion-aware applications in customer service, mental health, and interactive AI. Future enhancements can transform it into a fully context-aware, real-time emotion AI platform.

---
