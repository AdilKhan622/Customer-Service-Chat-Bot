# Customer Service Chatbot (NLP-Based)

A Python-based **Customer Service Chatbot** built using **Natural Language Processing (NLP)** techniques and a **Neural Network classifier**.  
The chatbot classifies user queries into predefined intents and responds with appropriate answers through an interactive **GUI interface**.

## Features

- Intent-based conversational chatbot
- NLP preprocessing using **NLTK**
- Bag-of-Words (BoW) feature representation
- Neural Network classifier using **TensorFlow / Keras**
- Interactive **Tkinter GUI**
- Confidence thresholding to handle unknown queries
- Modular and extensible design

## System Architecture

User Input -> Tokenization & Lemmatization (NLTK) -> Bag-of-Words Vectorization -> Neural Network Classifier (Keras) -> Intent Prediction -> Response Selection -> GUI Output


---

## ⚙️ Installation & Setup

### 1. Create and activate Conda environment
conda create -n chatbot python=3.10

conda activate chatbot

### 2. Install required dependencies
pip install tensorflow nltk numpy

### 3. Download NLTK resources
import nltk
nltk.download("punkt")
nltk.download("wordnet")

## Author 
Adil Ahmed Khan

Electronics and Computer Engineering

Customer Service Chatbot - Mini NLP project

