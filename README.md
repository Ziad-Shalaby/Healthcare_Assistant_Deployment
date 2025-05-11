# 🩺 Healthcare Assistant Deployment

An AI-powered Streamlit application that simulates a healthcare assistant by asking users symptom-related questions and predicting potential diseases using an ensemble of machine learning and deep learning models.

> Developed by [Ziad Shalaby](https://github.com/Ziad-Shalaby) and [Adham Nabih](https://github.com/ADHAM2nabih)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Features](#features)

---

## Project Overview

This Streamlit application aims to assist users with healthcare-related information in an intuitive and accessible way. By employing natural language processing (NLP) models, the assistant can understand and respond to user inputs, simulating a virtual consultation. The backend is structured to separate different aspects of its functionality, enhancing maintainability and scalability.

---

## Dataset

The application utilizes multiple healthcare-related datasets, each containing symptom and disease mappings. These datasets are preprocessed and tailored for specific models, including BiLSTMs and traditional classifiers.

---

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ziad-Shalaby/Healthcare_Assistant_Deployment.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Healthcare_Assistant_Deployment
   ```

3. **Create a virtual environment:**

   ```bash
   python -m venv env
   ```
5. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Activate the virtual environment:**
   
    - On Windows:

     ```bash
     env\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source env/bin/activate
     ```

## Usage

1. **Run the Jupyter Notebook:**

   Launch the notebook server:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the assistant:**

   -The assistant will ask a series of questions based on your initial symptom.
   -After collecting responses, it runs predictions using multiple models.
   -The final output is based on a majority voting system across models.

## Features

- **Symptom Question Generator:** Integrates with a public medical API to ask relevant questions.
- **Multi-Model Prediction System:** Utilizes various models trained on different datasets.
- **Dataset-Specific Tokenizers:** Stored in a /tokenizer folder to ensure accurate vectorization.
- **Voting Mechanism:** Aggregates model predictions to enhance reliability.
- **Streamlit UI:** Provides a user-friendly interface for easy interaction.

---

Let me know if you'd like assistance generating a `requirements.txt` file or any other documentation for your project.
::contentReference[oaicite:4]{index=4}
