<div align="center">

# INTELLIMENTAL

#### Multi-Modal AI System for Mental Health Analysis

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white" alt="Flutter"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black" alt="Firebase"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#ai-methodology">AI Methodology</a> ·
  <a href="#tech-stack">Tech Stack</a> ·
  <a href="#installation">Installation</a>
</p>

</div>

<br/>

## Overview

INTELLIMENTAL is an advanced, production-grade AI system designed to analyze mental health indicators through multi-modal data fusion. By synthesizing biometric patterns from facial expressions and linguistic cues from textual input, the system provides a robust, non-clinical initial assessment mechanism.

**Architecture Philosophy**

The system prioritizes low-latency inference and high scalability through a decoupled backend service architecture and cross-platform mobile client implementation.

**Key Capabilities**

- **Dual-Stream Processing**: Simultaneous analysis of visual and textual data streams
- **Late Fusion Strategy**: Aggregates independent probability distributions to minimize false positives
- **Edge-Optimized**: Mobile client designed for efficient asynchronous data transmission
- **Secure Infrastructure**: Integrated authentication and real-time database management via Firebase

<br/>

## Architecture

The system implements a modular microservices-patterned architecture, ensuring strict separation of concerns between the presentation layer, inference engine, and data persistence.

<table>
<thead>
<tr>
<th align="left">Component</th>
<th align="left">Technology</th>
<th align="left">Responsibility</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Client Layer</b></td>
<td>Flutter (Dart)</td>
<td>UI/UX, Camera Stream Management, Asynchronous API Calls</td>
</tr>
<tr style="background-color: #f6f8fa;">
<td><b>Orchestration</b></td>
<td>Flask, Gunicorn</td>
<td>Request Handling, Model Loading, JSON Serialization</td>
</tr>
<tr>
<td><b>Inference Engine</b></td>
<td>PyTorch, TensorFlow, Keras,OpenCV</td>
<td>Tensor Computation, Feature Extraction, Classification</td>
</tr>
<tr style="background-color: #f6f8fa;">
<td><b>Data Layer</b></td>
<td>Firebase Firestore</td>
<td>User Identity Management, Historical Record Storage</td>
</tr>
</tbody>
</table>

<br/>

## AI Methodology

### 1. Natural Language Processing (NLP)

Utilizes a Transformer-based architecture to capture long-range contextual dependencies in user journals.

```
Model: DeBERTa-v3-base
Technique: Disentangled Attention Mechanism
Framework: PyTorch, HuggingFace Transformers
Training Data: 65,000+ psychiatric text samples
Performance: F1-Score 0.88 (validation set)
```

### 2. Computer Vision (CV)

Employs efficient convolutional neural networks for real-time facial emotion recognition.

```
Backbone: EfficientNetB0 (Transfer Learning)
Framework: TensorFlow, Keras
Face Detection: YuNet (ONNX format for speed optimization)
Preprocessing: Albumentations (spatial & pixel-level augmentation)
Pipeline: OpenCV for image processing
```

### 3. Fusion Logic

The system applies a **Late Fusion** approach where probability logits from both NLP and CV models are extracted independently and passed through a weighted voting algorithm to produce the final diagnostic indicator.

**Benefits**
- Cross-modality verification reduces false positive rates
- Independent feature extraction preserves model-specific strengths
- Weighted ensemble improves overall prediction confidence

<br/>

## Tech Stack

<table>
<tr>
<td><b>Domain</b></td>
<td><b>Technologies & Libraries</b></td>
</tr>
<tr>
<td><b>Core AI & NLP</b></td>
<td>PyTorch · HuggingFace Transformers · DeBERTa</td>
</tr>
<tr>
<td><b>Computer Vision</b></td>
<td>TensorFlow · Keras · OpenCV · Albumentations · YuNet</td>
</tr>
<tr>
<td><b>Backend API</b></td>
<td>Flask · Gunicorn · Python · NumPy</td>
</tr>
<tr>
<td><b>Mobile Client</b></td>
<td>Flutter · Dart · Firebase</td>
</tr>
</table>

<br/>

## Repository Structure

```
intellimental/
├── data_preparation_and_training/    # Model training pipelines and research
├── backend/                          # Flask API and model serving infrastructure
├── intellimental/                    # Flutter mobile application
└── README.md                         # Project documentation
```

<br/>

## Installation

**Backend Setup**
```bash
git clone https://github.com/yourusername/intellimental.git
cd intellimental/backend
pip install -r requirements.txt
python app.py
```

**Mobile Client**
```bash
cd intellimental
flutter pub get
flutter run
```

<br/>

## Disclaimer

This system is designed for awareness and self-reflection purposes only. It is **not a clinical diagnostic tool** and should not replace professional psychological consultation.


<div align="center">

**Yahya Abu Zahra**  
Computer Engineering · Graduation Project 2026

</div>
