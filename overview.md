## 🔍 Overview

**RumorNet** is a smart web application that detects whether a given piece of news or social media text is **real or fake**. Built with **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, it helps tackle misinformation in real time.

This project is deployed using **Gradio UI** and hosted on **Hugging Face Spaces** for public access.

---

## 📌 Features

- 🔍 Text-based fake rumor classification
- 🧠 Supports traditional ML and transformer models (e.g. BERT, RoBERTa)
- 📊 Confidence scores for predictions
- 💡 Simple Gradio interface for easy interaction
- 🌐 Hosted live on Hugging Face Spaces

---

## ⚙️ Tech Stack

| Component            | Tech Used                           |
|---------------------|-------------------------------------|
| Language            | Python 3.10+                        |
| ML/NLP Libraries    | Scikit-learn, Transformers (BERT)   |
| Web UI              | Gradio                              |
| Hosting             | Hugging Face Spaces                 |

---

## 🧪 Model Training

We trained the model on a fake news dataset using the following steps:

- Text cleaning (stopword removal, lemmatization)
- TF-IDF / BERT embeddings
- Model: Logistic Regression / BERT fine-tuned
- Evaluation using accuracy, precision, recall




# RumorNet - AI-Based Fake Rumor Detection

🔍 Real-time NLP system to classify rumors as fake or real using BERT transformers.

## Features
- 85% accuracy on test dataset
- Real-time analysis with confidence scores
- Detailed explanations for predictions
- Built with BERT and Gradio

## Try it out!
Enter any statement to get instant fake/real classification.