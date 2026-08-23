# RTISLT
Real time Indian Sign Language Translation (dynamic)

## Project Pipeline

```text
ISL Video
    ↓
MediaPipe Holistic
    ↓
Hand + Pose Landmarks
    ↓
Normalized Landmark Sequence
    ↓
BiLSTM / Transformer
    ↓
ISL Sign / Gloss Sequence
    ↓
Seq2Seq / T5 Translation
    ↓
English Text
    ↓
Intent Classification
```

## Dataset Strategy

### INCLUDE-50 — Baseline

Used for isolated ISL sign recognition.

- 50 classes
- 958 videos
- 689 training
- 77 validation
- 192 test

Pipeline:

```text
INCLUDE Video
    ↓
MediaPipe
    ↓
225 Features / Frame
    ↓
Normalization
    ↓
.npy Sequence
    ↓
BiLSTM
    ↓
50-Class Prediction
```

### iSign — Main Translation Stage

Used after the isolated-sign baseline for continuous ISL processing and ISL-to-English translation.

```text
Continuous ISL
    ↓
Sequence Representation
    ↓
Translation Model
    ↓
English Sentence
```

## Landmark Representation

Each video frame currently uses:

```text
Left hand:   21 × 3 = 63
Right hand:  21 × 3 = 63
Pose:        33 × 3 = 99
-------------------------
Total:                   225
```

A video with `T` frames becomes:

```text
(T, 225)
```

Sequences are variable-length.

## Project Structure

```text
ISLR/
├── src/
│   ├── preprocessing/
│   │   ├── test_mediapipe.py
│   │   ├── normalize_landmarks.py
│   │   ├── record_dataset.py
│   │   ├── create_include_manifest.py
│   │   ├── verify_include.py
│   │   ├── download_include.py
│   │   └── preprocess_include.py
│   │
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── app/
│
├── data/
│   ├── include/
│   │   └── metadata/
│   │       └── include50_manifest.csv
│   └── landmarks/
│
├── models/
├── notebooks/
├── requirements.txt
└── README.md
```

The large INCLUDE dataset is stored separately on the local D: drive.

## Current Status

- [x] MediaPipe environment setup
- [x] Webcam landmark extraction
- [x] 225-feature representation
- [x] Landmark normalization
- [x] Custom recording pipeline
- [x] INCLUDE-50 metadata and manifest
- [x] 958/958 INCLUDE-50 videos verified
- [x] INCLUDE videos downloaded and extracted
- [ ] INCLUDE → MediaPipe preprocessing
- [ ] BiLSTM baseline
- [ ] Evaluation
- [ ] iSign continuous translation
- [ ] Intent classification
- [ ] FastAPI + Streamlit integration

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy / Pandas
- PyTorch
- Hugging Face Transformers
- FastAPI
- Streamlit

## Development Approach

The system is built incrementally:

```text
Video → Landmarks
       ↓
Landmarks → Isolated Sign
       ↓
Continuous Signs → Sequence
       ↓
Sequence → English
       ↓
English → Intent
```

This allows each stage to be tested independently before integration.
