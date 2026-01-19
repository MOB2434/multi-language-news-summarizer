from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import os
from englishsummarizer.summary import EnglishSummarizer
from nepalisummarizer.summary import NepaliSummarizer
from hindisummarizer.summary import HindiSummarizer

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
        data = request.json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        text = data.get('text', '')
        url = data.get('url', '')
        language = data.get('language', 'english').lower()
        num_sentences = int(data.get('num_sentences', 5))
        
        if not text and not url:
            return jsonify({
                'success': False,
                'error': 'Either text or URL must be provided'
            }), 400
        
        summarizer = None
        if language == 'english':
            summarizer = english_summarizer
        elif language == 'nepali':
            summarizer = nepali_summarizer
        elif language == 'hindi':
            summarizer = hindi_summarizer
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported language.'
            }), 400
        
        if not summarizer:
            return jsonify({
                'success': False,
                'error': f'{language.capitalize()} summarizer not initialized'
            }), 500
        
        summary = ""
        title = ""
        
        if url:

            logger.info(f"Summarizing URL: {url} in {language}")
            if hasattr(summarizer, 'summarize_url'):
                result = summarizer.summarize_url(url, num_sentences)
                if isinstance(result, tuple):
                    title, summary = result
                else:
                    summary = result
            else:
                title, content = summarizer.fetch_article(url)
                summary = summarizer.summarize(content, num_sentences)
        else:
            logger.info(f"Summarizing text in {language} ({len(text)} chars)")
            summary = summarizer.summarize(text, num_sentences)
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)