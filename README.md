# Face Emotion Recognition App

A Flask web app that captures a frame from your webcam and runs it through a MobileNetV2-based Keras model trained to classify facial expressions into seven emotions.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?style=flat-square&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0-000000?style=flat-square&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

<!-- TODO: add screenshot -->

## Overview

The project pairs a transfer-learning image classifier with a minimal web frontend. The browser streams the webcam into a `<video>` element, grabs a frame on click, and posts it to a Flask endpoint that preprocesses the image with OpenCV and runs inference with the bundled model (`FerModel1K.keras`).

> **Status: prototype.** The model is loaded and `model.predict` runs on every request, but `/predict` currently returns a fixed `"Happy"` label instead of mapping the predicted class index to an emotion name. Wiring up the label mapping is the next step.

## Key features

- In-browser webcam capture using `getUserMedia` and a hidden canvas
- Image upload to the backend as `multipart/form-data`
- Server-side preprocessing: decode, resize to 224×224, scale to [0, 1]
- Keras inference with a seven-class softmax output
- `Procfile` included for Heroku-style deployment

## Tech stack

| Layer | Technology |
|---|---|
| Model | TensorFlow 2.16 / Keras 3, MobileNetV2 backbone |
| Backend | Flask 2.0, OpenCV, NumPy |
| Frontend | Plain HTML + JavaScript (MediaDevices API, Canvas, Fetch) |

## Model architecture

Read from the saved model's config (`FerModel1K.keras`, saved with Keras 3.3.3):

```
Input (224 × 224 × 3)
  → MobileNetV2 feature extractor (inverted residual blocks, 1280-channel output)
  → GlobalAveragePooling2D
  → Dense(128, ReLU)
  → Dense(64, ReLU)
  → Dense(7, softmax)
```

- Compiled with the Adam optimizer, `sparse_categorical_crossentropy` loss and accuracy as the metric.
- Seven output classes, the same count as common facial-expression datasets such as FER-2013. The training notebook, dataset and class order are not included in this repo.

## Getting started

### Prerequisites

- Python 3.9–3.12 (required by TensorFlow 2.16)
- A webcam and a browser that allows camera access on `localhost`

### Install and run

```bash
git clone https://github.com/lakygosh/FaceEmotionRecognitionApp.git
cd FaceEmotionRecognitionApp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, allow camera access, and press **Capture**. The result appears in a browser alert.

No environment variables are needed. The model file (~30 MB) is loaded from the project root at startup.

## Project structure

```
FaceEmotionRecognitionApp/
├── app.py               # Flask app: serves the page and the /predict endpoint
├── FerModel1K.keras     # Trained Keras model (MobileNetV2 + dense head)
├── templates/
│   └── index.html       # Webcam capture UI
├── requirements.txt
└── Procfile             # web: python app.py
```

## Author

Lazar Gošić — GitHub [@lakygosh](https://github.com/lakygosh)
