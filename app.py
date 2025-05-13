import streamlit as st
import requests
import joblib
import numpy as np
import re
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')

# ========== Page Style ==========
from streamlit.components.v1 import html
st.set_page_config(page_title="AI Healthcare Assistant", layout="centered")

# ========== Custom CSS for dark theme ==========
st.markdown(
    """
    <style>
    .stApp { background-color: #121212; color: #f0f0f0; }
    h1, h2, h3, h4, h5, h6, p, label, div, span { color: #e0e0e0 !important; }
    .stTextInput>div>div>input {
        background-color: #1f1f1f !important;
        border: 1px solid #333 !important;
        color: #ffffff !important;
    }
    [data-testid="stButton"] button {
        background-color: #3f51b5 !important;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    [data-testid="stButton"] button:hover {
        background-color: #5c6bc0 !important;
    }
    .stSuccess, .stError, .stInfo {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Processor Class ==========
class Processor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        return re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        return ' '.join([word for word in words if word.lower() not in self.stop_words])

    def lemmatize_text(self, text):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return self.lemmatize_text(text)

processor = Processor()

# ========== Load Models, Encoders, Tokenizers ==========
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

@st.cache_resource
def load_encoder(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading encoder from {path}: {e}")
        return None

@st.cache_resource
def load_tokenizer(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading tokenizer from {path}: {e}")
        return None

model_paths = {
    '1a': 'machine_learning_models/1-log_reg_model.pkl',
    '1b': 'machine_learning_models/1-mlp_model.pkl',
    '1c': 'deep_learning_models/1-lstm_model.pkl',
    '2a': 'machine_learning_models/2-logistic_model.pkl',
    '2b': 'machine_learning_models/2-mlp_model.pkl',
    '3a': 'machine_learning_models/3-log_reg_model.pkl',
    '3b': 'machine_learning_models/3-mlp_model.pkl',
    '3c': 'deep_learning_models/3-lstm_model.pkl',
}

encoder_paths = {
    '1': 'label_encoder/1-label_encoder.pkl',
    '2': 'label_encoder/2-label_encoder.pkl',
    '3': 'label_encoder/3-label_encoder.pkl',
}

tokenizer_paths = {
    '1c': 'tokenizer/1-tokenizer.pkl',
    '3c': 'tokenizer/3-tokenizer.pkl'
}

models = {key: load_model(model_paths[key]) for key in model_paths}
encoders = {key: load_encoder(encoder_paths[key]) for key in encoder_paths}
tokenizers = {key: load_tokenizer(tokenizer_paths[key]) for key in tokenizer_paths}

model_type_mapping = {
    '1a': 'Logistic Regression',
    '1b': 'MLP',
    '1c': 'LSTM',
    '2a': 'Logistic Regression',
    '2b': 'MLP',
    '3a': 'Logistic Regression',
    '3b': 'MLP',
    '3c': 'LSTM',
}

# ========== Preprocessing Functions ==========
def preprocess_text_for_ml(text):
    return processor.preprocess(text)

def preprocess_text_for_lstm(text):
    return processor.preprocess(text)

# ========== Question Generation API ==========
def get_next_question(qa_history):
    conversation = ""
    for i in range(1, len(qa_history), 2):
        q = qa_history[i - 1]
        a = qa_history[i]
        conversation += f"Q{i//2+1}: {q}\nA{i//2+1}: {a}\n"

    prompt = f"""
You are a healthcare assistant. Based on the previous questions and answers, ask a new, relevant, and **different** question to help diagnose the problem. Do not repeat any earlier questions.

Previous Q&A:
{conversation}

Next question:
"""
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['general']['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo-0125",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ Error getting question: {str(e)}"

# ========== Specialist Recommendation API ==========
def get_specialist_recommendation(user_description):
    prompt = f"""
A user has described their symptoms as follows:
\"{user_description}\"

Based on these symptoms, write a short report (2 sentences max) in English:
1. Recommend a medical specialty the user should visit.
2. Give one brief health tip they can follow until they see a doctor.
"""
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['general']['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo-0125",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ Error getting specialist advice: {str(e)}"

# ========== Streamlit App ==========
st.title("🧠 AI Healthcare Assistant")

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.qa_pairs = []
    st.session_state.valid_qa_pairs = []
    st.session_state.max_qs = 6

if st.session_state.step == 0:
    symptoms = st.text_input("What symptoms are you experiencing? (Example: headache, dizziness)")
    if st.button("Start Diagnosis") and symptoms:
        st.session_state.qa_pairs.append("What symptoms are you experiencing?")
        st.session_state.qa_pairs.append(symptoms)
        st.session_state.valid_qa_pairs.append("What symptoms are you experiencing?")
        st.session_state.valid_qa_pairs.append(symptoms)
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step <= st.session_state.max_qs:
    if len(st.session_state.qa_pairs) % 2 == 0:
        question = get_next_question(st.session_state.qa_pairs)
        st.session_state.qa_pairs.append(question)

    st.subheader(f"Question {st.session_state.step}:")
    current_question = st.session_state.qa_pairs[-1]
    st.write(current_question)
    answer = st.text_input(f"Your answer to question {st.session_state.step}:")

    if st.button("Next") and answer:
        st.session_state.qa_pairs.append(answer)
        if not any(neg in answer.lower() for neg in ["no", "not sure", "don't have"]):
            st.session_state.valid_qa_pairs.append(current_question)
            st.session_state.valid_qa_pairs.append(answer)
        st.session_state.step += 1
        st.rerun()

else:
    st.success("✅ The questions are complete. Analyzing your health status now...")
    qa_text = [f"{st.session_state.valid_qa_pairs[i]}: {st.session_state.valid_qa_pairs[i+1]}"
               for i in range(0, len(st.session_state.valid_qa_pairs), 2)]
    input_data = [" ".join(qa_text)]

    all_predictions = []
    for key, model in models.items():
        if model is not None:
            try:
                model_type = model_type_mapping[key]
                processed_input = [preprocess_text_for_ml(text) if model_type != 'LSTM' else preprocess_text_for_lstm(text)
                                   for text in input_data]

                if model_type == 'LSTM':
                    tokenizer = tokenizers.get(key)
                    if tokenizer:
                        seq = tokenizer.texts_to_sequences(processed_input)
                        processed_input = pad_sequences(seq, maxlen=50)
                    else:
                        raise ValueError("Tokenizer not found.")

                pred = model.predict(processed_input)
                if hasattr(pred, "shape") and len(pred.shape) > 1:
                    pred = np.argmax(pred, axis=1)

                encoder_key = key[0]
                encoder = encoders.get(encoder_key)
                if encoder:
                    disease = encoder.inverse_transform(pred)[0]
                    all_predictions.append(disease)
                else:
                    st.error(f"Encoder not found for model {key}")
            except Exception as e:
                st.error(f"An error occurred during prediction with model {key}: {str(e)}")

    if all_predictions:
        most_common_prediction, _ = Counter(all_predictions).most_common(1)[0]
        st.subheader(f"🔍 Predicted Disease: {most_common_prediction}")
        recommendation = get_specialist_recommendation(input_data[0])
        st.subheader("🩺 Specialist Recommendation:")
        st.write(recommendation)
        st.info("💡 Temporary advice: Please rest and drink plenty of fluids until you visit a doctor.")
    else:
        st.error("No predictions were made from any model.")

    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
