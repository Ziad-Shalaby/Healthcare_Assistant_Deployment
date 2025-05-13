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
6. [Project Structure](#project-structure)
7. [Technologies](#technologies)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview
This Streamlit application aims to assist users with healthcare-related information in an intuitive and accessible way. By employing natural language processing (NLP) models, the assistant can understand and respond to user inputs, simulating a virtual consultation. The backend is structured to separate different aspects of its functionality, enhancing maintainability and scalability.

## Dataset
The application utilizes multiple healthcare-related datasets, each containing symptom and disease mappings. These datasets are preprocessed and tailored for specific models, including:
- BiLSTM Neural Networks
- Traditional Machine Learning Classifiers
- Ensemble Learning Models

### Data Preprocessing
- Custom tokenization
- Feature engineering
- Data cleaning and normalization

## Installation

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ziad-Shalaby/Healthcare_Assistant_Deployment.git
   cd Healthcare_Assistant_Deployment
   ```

2. **Create a virtual environment:**
   ```bash
   # On Windows
   python -m venv env
   env\Scripts\activate

   # On macOS/Linux
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Interaction Flow
1. The assistant initiates with an initial symptom inquiry
2. Asks a series of follow-up questions based on user responses
3. Processes input through multiple machine learning models
4. Provides a prediction based on ensemble voting mechanism

## Features
- 🤖 **Intelligent Symptom Analysis**
  - Natural Language Processing
  - Context-aware questioning
  - Multi-model prediction system

- 📊 **Advanced Modeling**
  - Ensemble learning approach
  - Multiple model integration
  - Majority voting prediction mechanism

- 🧠 **Model Capabilities**
  - Symptom Question Generator
  - Dataset-specific tokenizers
  - Robust prediction mechanism

## Technologies
- **Programming**: Python 3.8+
- **Machine Learning**:
  - Scikit-learn
  - TensorFlow
  - Keras
- **NLP**:
  - NLTK
  - SpaCy
- **Web Framework**:
  - Streamlit
- **Data Processing**:
  - Pandas
  - NumPy

## Contributing
1. Fork the repository
2. Create your feature branch 
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes 
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. Push to the branch 
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
- Ziad Shalaby - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/ziad-shalaby1/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Ziad-Shalaby)
- Adham Nabih - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/adham-nabih/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/ADHAM2nabih)
---

> 💡 **Disclaimer**: This AI-assisted tool is for educational purposes and should not replace professional medical advice. Always consult healthcare professionals for accurate diagnosis and treatment.
