import streamlit as st
from transformers import pipeline
import fitz  
import os
import re
from langdetect import detect
from googletrans import Translator
import requests

st.set_page_config(page_title="Multilingual Text Summarizer", page_icon="📝", layout="wide")

# API URLs and headers
API_URL_Facebook_BART = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_URL_Distil_Bert = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
API_URL_Samsum_Bert = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
API_URL_Google_Pegasus = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
API_URL_Pegasus_CNN_Daily_Mail = "https://api-inference.huggingface.co/models/google/pegasus-cnn_dailymail"
API_URL_LLAMA = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-405B"
headers = {"Authorization": "Bearer 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"}

@st.cache_resource
def load_model(api_url, payload):
    response = requests.post(api_url, headers=headers, json=payload)
    response_data = response.json()
    if isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict):
        return response_data[0].get('summary_text', "No summary found")
    return "Invalid response format."

def summarize_text(text, model_type, summary_length, custom_prompt=None):
    cleaned_text = preprocess_text(text)
    if custom_prompt:
        cleaned_text = f"{custom_prompt}. {cleaned_text}"

    payload = {"inputs": cleaned_text, "parameters": {"max_length": summary_length, "min_length": summary_length - 30}}  # Adjust as needed

    if model_type == "Falcons AI":
        summary = load_model("Falconsai/text_summarization", payload)

    elif model_type == "Distil-BERT":
        summary = load_model(API_URL_Distil_Bert, payload)

    elif model_type == "Pegasus Google":
        summary = load_model(API_URL_Google_Pegasus, payload)

    elif model_type == "Pegasus CNN":
        summary = load_model(API_URL_Pegasus_CNN_Daily_Mail, payload)

    elif model_type == "BART-Samsum":
        summary = load_model(API_URL_Samsum_Bert, payload)

    elif model_type == "Facebook-BART":
        summary = load_model(API_URL_Facebook_BART, payload)

    elif model_type == "Meta-Llama":
        summary = load_model(API_URL_LLAMA, payload)

    else:
        summary = "Invalid model type specified."

    return summary

def preprocess_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def count_words(text):
    return len(text.split())

# Streamlit UI
st.title("📝 SummariX")
st.write("**Welcome to the multilingual text summarizer!** Enter your text directly or upload a text/PDF file below.")

st.sidebar.write("### Input Method")
input_method = st.sidebar.radio("Choose input method:", ("Direct Text Input", "Upload File (PDF, TXT)"))

st.sidebar.write("### Summary Length")
summary_length = st.sidebar.slider("Select Summary Length (words):", min_value=50, max_value=200, value=100, step=10)
select_all_models = st.sidebar.checkbox("Select All Models")

models = ["Meta-Llama", "Falcons AI", "BART-Samsum", "Distil-BERT", "Pegasus Google", "Pegasus CNN", "Facebook-BART"]
selected_models = []
for idx, model in enumerate(models):
    if select_all_models:
        selected_models.append(model)
        st.sidebar.checkbox(model, value=True, key=f"checkbox_{idx}")
    else:
        if st.sidebar.checkbox(model, key=f"checkbox_{idx}"):
            selected_models.append(model)

if input_method == "Direct Text Input":
    user_input = st.text_area("Enter your text here:", height=200)
else:
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            file_text = read_pdf(uploaded_file)
        elif file_extension == ".txt":
            file_text = read_txt(uploaded_file)
        else:
            file_text = None
            st.error("Unsupported file type. Please upload a PDF or TXT file.")

# Add a section for custom prompts
st.write("### Refine your summary:")
custom_prompt = st.text_input("Enter a prompt to refine the summary, e.g., 'focus on key points, remove any unnecessary details from the summary.'")

if 'file_text' not in locals():
    file_text = user_input if input_method == "Direct Text Input" else None

# Initialize the Translator
translator = Translator()

if file_text:
    input_word_count = count_words(file_text)
    st.write(f"**Token Count of Input Text:** {input_word_count}")

    # Detect language
    try:
        detected_lang = detect(file_text)
        st.markdown(f"**Detected Language:** {detected_lang.upper()}", unsafe_allow_html=True)  # Highlighted language display
        if detected_lang in ['it', 'es']:  # Italian or Spanish
            st.session_state.translated_text = translator.translate(file_text, src=detected_lang, dest='en').text
            st.success("Translation successful!")
            # Checkbox to display the translated text
            show_translated = st.checkbox("Show Translated Text", value=True)
            if show_translated:
                st.write(f"**Translated Text:** {st.session_state.translated_text}")
            translated_text = st.session_state.translated_text
        else:
            translated_text = file_text  # Default to original for other languages

    except Exception as e:
        st.error(f"Error detecting language: {e}")

    # Option for summary language
    summary_language = st.radio("Select Summary Language:", ("Original Language", "English"))

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summaries = {}
                # Use translated text for summarization
                for model in selected_models:
                    # Use original text if user chooses original language, else use translated text
                    summary_input = translated_text if summary_language == "English" else file_text
                    summary = summarize_text(summary_input, model, summary_length, custom_prompt)
                    summaries[model] = summary

                st.subheader("Summaries")

                st.markdown("""
                <style>
                .summary-block {
                    background-color: #f0f0f0;
                    border-radius: 15px;
                    padding: 20px;
                    margin: 10px 0;
                    height: 100%;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                </style>
                """, unsafe_allow_html=True)

                num_models = len(summaries)
                for i in range(0, num_models, 3):
                    cols = st.columns(min(3, num_models - i))

                    for j in range(min(3, num_models - i)):
                        model_name = list(summaries.keys())[i + j]
                        summary = summaries[model_name]
                        summary_word_count = count_words(summary)

                        with cols[j]:
                            st.markdown(f"""
                            <div class="summary-block">
                                <strong>{model_name}:</strong><br>{summary}<br>
                                <strong>Word Count:</strong> {summary_word_count}
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.write("Please enter some text or upload a file to get started.")
