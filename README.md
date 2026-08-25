# RTISLT
# ISL Translator

A lightweight, landmark-based Indian Sign Language (ISL) to English translation system focused on **manual + non-manual linguistic information**, signer-independent evaluation, and efficient inference.

---

## Proposed Architecture

```text
                         RAW ISL VIDEO / WEBCAM
                                   │
                                   ▼
                         MediaPipe Holistic
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
        Hand landmarks       Body/Pose landmarks    Face/Head landmarks
              │                    │                    │
              └──────────────┐     │     ┌──────────────┘
                             ▼     ▼     ▼
                       ┌─────────────────────┐
                       │   Feature Streams   │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Manual Encoder              NMF Encoder
          (hands + body pose)        (face + head cues)
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         Cross-Stream Fusion
                                  │
                                  ▼
                     Local Temporal Encoder
                                  │
                                  ▼
                      Global Transformer
                           Encoder
                                  │
                                  ▼
                   Transformer English Decoder
                                  │
                                  ▼
                         English Sentence
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                Intent Classifier     NMF Auxiliary
                  (DistilBERT)          Prediction

Core Idea

The system accepts raw video/webcam input, but the trained translation model operates on structured landmarks rather than raw RGB pixels.

Raw video
   ↓
MediaPipe
   ↓
Hand + body + face/head landmarks
   ↓
Manual / NMF feature streams
   ↓
Transformer-based translation
   ↓
English sentence
Research Gaps Targeted
1. Explicit Non-Manual Feature Modeling

Recent systems can already use face/body keypoints, so the claim is not that facial information has never been used.

Our target is the explicit modeling of linguistically meaningful non-manual features (NMFs) as a dedicated branch, followed by controlled ablation and, where labels are available, auxiliary supervision.

2. Signer-Independent Generalization

We will explicitly compare standard evaluation with unseen-signer evaluation.

Standard split
      vs
Signer-independent split

This tests whether the model generalizes beyond the people seen during training.

3. Local + Global Temporal Modeling

Sign language contains both short, fine-grained movements and longer sentence-level context.

The proposed model therefore uses:

Landmark sequence
      ↓
Local Temporal Encoder
      ↓
Global Transformer
      ↓
English Decoder
4. Translation Quality

Continuous ISL-to-English translation remains challenging. The objective is to improve translation through better representation and temporal/linguistic modeling rather than simply using a larger video model.

5. Efficiency

Because the model uses landmarks instead of raw RGB features, we can evaluate the trade-off between:

translation quality
parameter count
latency
FPS
memory
Dataset Strategy
INCLUDE-50 — Baseline

Used for isolated-sign recognition.

Classes:       50
Train:        689
Validation:    77
Test:         192
Total:        958

Baseline:

INCLUDE video
      ↓
MediaPipe
      ↓
225 landmark features/frame
      ↓
Normalization
      ↓
BiLSTM
      ↓
50-class prediction

The BiLSTM is a baseline, not the main proposed contribution.

iSign — Main Translation Dataset

Used for continuous ISL-to-English translation.

iSign video / pose
       ↓
Landmark representation
       ↓
Manual + NMF streams
       ↓
Proposed Transformer
       ↓
English sentence
Landmark Representation

Current baseline:

Left hand:    21 × 3 = 63
Right hand:   21 × 3 = 63
Body pose:    33 × 3 = 99
--------------------------------
Total:                  225 features/frame

A video with T frames becomes:

(T, 225)

The face/head representation will be designed separately for the NMF branch instead of adding the entire face mesh directly to the baseline vector.

Model Components
Baseline

BiLSTM

Used for isolated sign recognition on INCLUDE-50.

Proposed Model

Transformer Encoder–Decoder

Manual encoder: hand + body/pose features
NMF encoder: face + head features
Cross-stream feature fusion
Local temporal encoder
Global Transformer encoder
English Transformer decoder
Optional Auxiliary Branch

NMF prediction head

Used when reliable non-manual labels are available.

Downstream

DistilBERT

Used for intent classification from translated English.

Experimental Plan
Experiment 1 — Isolated baseline
Hands + body → BiLSTM

Metrics:

Accuracy
Macro-F1
Precision
Recall
Confusion matrix
Experiment 2 — Pose Transformer baseline
Hands + body → Transformer → English
Experiment 3 — Face/head augmentation
Hands + body + face/head → Transformer
Experiment 4 — Dedicated NMF branch
Manual stream + NMF stream → Fusion → Transformer
Experiment 5 — NMF auxiliary supervision

Add an auxiliary objective for non-manual linguistic cues when reliable labels are available.

Experiment 6 — Local + global temporal modeling

Compare:

Standard Transformer
vs
Local Temporal Encoder + Global Transformer
Experiment 7 — Signer-independent evaluation

Evaluate the strongest models on previously unseen signers.

Experiment 8 — Efficiency

Measure:

model size
parameter count
memory
latency
FPS
BLEU / chrF / WER
Intended Contribution
A lightweight landmark-based continuous ISL-to-English architecture with explicit manual/non-manual feature separation.
A controlled study of whether non-manual cues improve translation.
Signer-independent evaluation of the proposed approach.
Accuracy-versus-efficiency analysis for practical deployment.
Proposed Contribution Statement

We propose a lightweight landmark-based framework for continuous Indian Sign Language-to-English translation that explicitly models manual and non-manual linguistic cues, fuses them through a temporal Transformer architecture, and evaluates the resulting system under signer-independent and efficiency-constrained conditions.

Implementation Roadmap
Phase 1
INCLUDE → MediaPipe → BiLSTM baseline

Phase 2
iSign pose → Transformer baseline

Phase 3
Add face/head stream

Phase 4
Add dedicated NMF encoder + auxiliary objective

Phase 5
Add local-to-global temporal modeling

Phase 6
Cross-signer evaluation

Phase 7
Efficiency optimization

Phase 8
FastAPI + Streamlit deployment
Tech Stack
Python
OpenCV
MediaPipe
NumPy
Pandas
PyTorch
Hugging Face Transformers
scikit-learn
FastAPI
Streamlit
Project Structure
ISLR/
├── src/
│   ├── preprocessing/
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

Large INCLUDE videos are stored separately on the local D: drive.

Current Status
 Python environment
 MediaPipe setup
 Webcam landmark extraction
 225-feature representation
 Landmark normalization
 Custom recording pipeline
 INCLUDE-50 metadata
 INCLUDE-50 manifest
 958/958 videos verified
 INCLUDE videos downloaded and extracted
 INCLUDE → MediaPipe preprocessing
 BiLSTM baseline
 iSign Transformer baseline
 Manual/NMF fusion model
 Signer-independent evaluation
 Efficiency evaluation
 FastAPI + Streamlit deployment
Terminology

Landmark: A coordinate describing a detected body, hand, face, or head point.

Landmark-based / pose-based: The model receives structured coordinates rather than raw RGB pixels.

Gloss: A dataset-defined textual label representing a sign.

Temporal encoder: A neural component that learns how landmark positions change over time.

Transformer encoder: Processes the input sequence and creates contextual representations.

Transformer decoder: Generates the English sentence token by token.

NMF: Non-Manual Features such as facial/head movements that can carry linguistic information.

Research Positioning

The project should not claim that pose-based ISL translation itself is novel.

The intended novelty is the combination of:

Manual landmark stream
        +
Dedicated non-manual stream
        +
Local-to-global temporal modeling
        +
Signer-independent evaluation
        +
Efficiency analysis

This positions the work as an experimentally testable extension of existing pose-based ISL translation rather than a simple reproduction.