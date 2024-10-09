# SummariX
📝 SummariX is a Multilingual Text Summarizer application that summarizes text, PDFs, and images in multiple languages.

## Features

- Summarize text input directly
- Summarize content from uploaded PDF, TXT, and image files
- Detect and handle multiple languages
- Translate summarized text to English
- Chat-like prompt system for refining summaries
- User can select the sigle or multiple models according to choice
- Display tokens used and word count of summary generated

## Demo

You can try the live demo on [Streamlit Cloud](https://llm-based-text-summarizer.streamlit.app/).

## LLM Models used
We used below Hugging Face open source models which are specifice to text summarization.
1. facebook/bart-large-cnn
2. meta-llama/Llama-3.1-405B
3. google/pegasus-xsum
4. distilbart-cnn-12-6
5. bart-large-cnn-samsum
6. google/pegasus-cnn_dailymail
7. T5 transformer Model



## Installation

### Local Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/prakashgore/SummariX.git
    cd LLM-Based-Text-Summarizer
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv myenv
    source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

### Docker

1. **Pull the Docker image:**

    ```sh
    docker pull yamin69/summarizer:latest
    ```

2. **Run the Docker container:**

    ```sh
    docker run -p 8501:8501 yamin69/summarizer:latest
    ```

## Usage

1. **Navigate to the app URL.**
2. **Choose an input method:**
    - Direct Text Input
    - Upload File (PDF, TXT, Image)
3. **Enter or upload your content.**
4. **Optionally add prompts to refine the summary.**
5. **Click "Generate Summary" to get the summarized text.**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)


