from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import os
from englishsummarizer.summary import EnglishSummarizer
from nepalisummarizer.summary import NepaliSummarizer
from hindisummarizer.summary import HindiSummarizer
import nltk

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://minorproject-taupe.vercel.app/"],
        "methods": ["GET", "POST","OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

english_summarizer = None
nepali_summarizer = None
hindi_summarizer = None

def ensure_nltk_resources():
    data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
    os.makedirs(data_dir, exist_ok=True)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'punkt_tab': 'tokenizers/punkt_tab'  
    }

    for name, path_id in resources.items():
        try:
            nltk.data.find(path_id)
            logger.debug(f"NLTK resource already available: {name}")
        except LookupError:
            logger.info(f"NLTK resource '{name}' not found — downloading to {data_dir}")
            try:
                nltk.download(name, download_dir=data_dir, quiet=True)
                nltk.data.find(path_id)
                logger.info(f"Downloaded NLTK resource: {name}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource '{name}': {e}")

ensure_nltk_resources()


def initialize_summarizers():
    global english_summarizer, nepali_summarizer, hindi_summarizer

    try:
        english_summarizer = EnglishSummarizer()
        nepali_summarizer = NepaliSummarizer()
        hindi_summarizer = HindiSummarizer()

        english_model_path = 'models/english_model.pkl'
        nepali_model_path = 'models/nepali_model.pkl'
        hindi_model_path = 'models/hindi_model.pkl'

        if os.path.exists(english_model_path):
            english_summarizer.load_model(english_model_path)
            logger.info("English summarizer model loaded successfully.")

        if os.path.exists(nepali_model_path):
            nepali_summarizer.load_model(nepali_model_path)
            logger.info("Nepali summarizer model loaded successfully.")

        if os.path.exists(hindi_model_path):
            hindi_summarizer.load_model(hindi_model_path)
            logger.info("Hindi summarizer model loaded successfully.")

    except Exception as e:
        logger.error(f"Error initializing summarizers: {e}")

initialize_summarizers()

@app.route('/')
def home():
    return jsonify({
        'message': 'Multi-language News Summarizer API',
        'status': 'running',
        'endpoints': {
            '/summarize': 'POST - Summarize text or URL',
            '/languages': 'GET - Get supported languages'
        }
    })

@app.route('/summarizer', methods=['POST'])
def summarize():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        text = data.get('text', '')
        url = data.get('url', '')
        language = data.get('language', 'english').lower()
        num_sentences = int(data.get('num_sentences', 5))

        if not text and not url:
            return jsonify({'success': False, 'error': 'Either text or URL must be provided'}), 400

        summarizer = None
        if language == 'english':
            summarizer = english_summarizer
        elif language == 'nepali':
            summarizer = nepali_summarizer
        elif language == 'hindi':
            summarizer = hindi_summarizer
        else:
            return jsonify({'success': False, 'error': 'Unsupported language.'}), 400

        if not summarizer:
            return jsonify({'success': False, 'error': f'{language.capitalize()} summarizer not initialized'}), 500

        title = ""
        summary = ""

        def get_summary_from_result(result):
            """Convert summarizer result to text if it returns a file path."""
            if isinstance(result, tuple):
                t, s = result
            else:
                t, s = "", result

            if isinstance(s, str) and os.path.exists(s):
                try:
                    with open(s, 'r', encoding='utf-8') as f:
                        s = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read summary file '{s}': {e}")

            return t, s

        if url:
            logger.info(f"Summarizing URL: {url} in {language}")
            if hasattr(summarizer, 'summarize_url'):
                result = summarizer.summarize_url(url, num_sentences)
                title, summary = get_summary_from_result(result)
            else:
                title, content = summarizer.fetch_article(url)
                result = summarizer.summarize(content, num_sentences)
                title, summary = get_summary_from_result(result)
        else:
            logger.info(f"Summarizing text in {language} ({len(text)} chars)")
            result = summarizer.summarize(text, num_sentences)
            title, summary = get_summary_from_result(result)

        response = {
            'success': True,
            'language': language,
            'summary': summary,
            'num_sentences_used': num_sentences
        }

        if title:
            response['title'] = title
        if url:
            response['url'] = url

        logger.info(f"Summary generated successfully for {language}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_languages():

    return jsonify({
        'languages': [
            {
                'code': 'en',
                'name': 'English',
                'supported': english_summarizer is not None
            },
            {
                'code': 'ne',
                'name': 'Nepali',
                'supported': nepali_summarizer is not None
            },
            {
                'code': 'hi',
                'name': 'Hindi',
                'supported': hindi_summarizer is not None
            }
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)