# 🩺 Healthcare Assistant App

A Streamlit-based AI healthcare assistant that intelligently asks symptom-related questions and predicts possible diseases using multiple machine learning and deep learning models.

> Developed by [Ziad Shalaby](https://github.com/Ziad-Shalaby) and [Adham Nabih](https://github.com/ADHAM2nabih)

---

## 🚀 Overview

The **Healthcare Assistant** app simulates a virtual consultation by asking users six symptom-based questions and predicting the most likely diagnosis using a voting system across multiple models. It combines NLP preprocessing, classification models, and an API-powered questioning system.

---

## 🧠 Key Features

- **Interactive Chat Interface:** Asks the user 6 follow-up health-related questions.
- **Symptom Question Generation:** Uses a public API to generate dynamic symptom-related questions.
- **Multi-Model Prediction:** Uses 8 trained models (Logistic Regression, Random Forest, BiLSTM, etc.) across 3 datasets.
- **Voting System:** Aggregates model predictions and selects the most frequent one.
- **Tokenizer Management:** Loads dataset-specific tokenizers from a dedicated folder.
- **Clean UI:** Built using Streamlit for a user-friendly interface.

---

## 🗂️ Project Structure

Healthcare_Assistant_Deployment/
│
├── app.py # Main Streamlit app
├── model_utils.py # Model loading and prediction logic
├── api_utils.py # API call functions for question generation
├── preprocess_utils.py # NLP preprocessing functions
├── tokenizer/ # Contains tokenizers for each dataset
├── models/ # Saved model files
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **TensorFlow / Keras**
- **NLTK / spaCy**
- **FastText embeddings**
- **Public Symptom API**

---

## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ziad-Shalaby/Healthcare_Assistant_Deployment.git
   cd Healthcare_Assistant_Deployment
2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**

   ```bash
   streamlit run app.py
   ```
   
