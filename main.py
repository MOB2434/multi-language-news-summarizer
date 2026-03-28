from flask import Flask, render_template, request, jsonify,url_for, redirect, flash
from flask_cors import CORS
import logging
from englishsummarizer.summary import EnglishSummarizer
from nepalisummarizer.summary import NepaliSummarizer
from hindisummarizer.summary import HindiSummarizer
import nltk
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import InputRequired, Length, ValidationError, DataRequired
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import translate
from datetime import datetime
from detector import FakeNewsDetector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://minorproject-taupe.vercel.app/"],
        "methods": ["GET", "POST","OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type"]
    }
})

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

@app.route('/')
def home():
   return render_template('home.html')

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

translator_tokenizer = None
translator_model = None

MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANGS = {
    "en": "eng_Latn",
    "ne": "npi_Deva",
    "hi": "hin_Deva",
    "mai": "mai_Deva",
    "sa": "san_Deva",
    "bho": "bho_Deva",
}

def initialize_translator():
    """Initialize the NLLB translator model"""
    global translator_tokenizer, translator_model
    try:
        logger.info("Loading NLLB translator model...")
        translator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        translator_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        logger.info("Translator model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading translator model: {e}")
        translator_tokenizer = None
        translator_model = None

initialize_translator()

detector = FakeNewsDetector()

@app.route('/summarizer', methods=['POST'])
def summarize():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        text = data.get('text', '')
        url = data.get('url', '')
        language = data.get('language', 'english').lower()
        num_sentences = int(data.get('num_sentences', 5))
        translate_to = data.get('translate_to', None) 
        
        if not text and not url:
            return jsonify({
                'success': False,
                'error': 'Either text or URL must be provided'
            }), 400
        
        lang_to_code = {
            'english': 'en',
            'nepali': 'ne',
            'hindi': 'hi'
        }

        summarizer = None
        if language == 'english':
            summarizer = english_summarizer
            src_lang_code = 'en'
        elif language == 'nepali':
            summarizer = nepali_summarizer
            src_lang_code = 'ne'
        elif language == 'hindi':
            summarizer = hindi_summarizer
            src_lang_code = 'hi'
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

        if url:
            news_text = detector.extract(url)
            news_result = detector.detect(news_text)
        else:
            news_result = detector.detect(text)

        
        response = {
            'success': True,
            'result': news_result,
            'language': language,
            'language_code': src_lang_code,
            'summary': summary,
            'num_sentences_used': num_sentences
        }
        
        if translate_to and translate_to in LANGS and translate_to != src_lang_code:
            try:
                logger.info(f"Translating summary from {src_lang_code} to {translate_to}")
                translated_summary = translate(summary, src_lang_code, translate_to)
                
                translated_title = None
                if title:
                    translated_title = translate(title, src_lang_code, translate_to)
                
                response['translated_summary'] = translated_summary
                response['translated_title'] = translated_title
                response['translated_to'] = translate_to
                
                logger.info(f"Translation completed successfully")

            except Exception as e:
                logger.error(f"Translation failed: {e}")
                response['translation_error'] = str(e)

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
        'summarization_languages': [
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
        ],

        'translation_languages': [
            {'code': code, 'name': name} for code, name in [
                ('en', 'English'),
                ('ne', 'Nepali'),
                ('hi', 'Hindi'),
                ('mai', 'Maithili'),
                ('sa', 'Sanskrit'),
                ('bho', 'Bhojpuri')
            ]
        ]
    })

load_dotenv()

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://user:password@hostname/database')

if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 60,
    'pool_pre_ping': True,
    'connect_args': {
        'sslmode': 'require' 
    }
}

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key-change-this')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    saved_news = db.relationship('SavedNews', backref='user', lazy=True, cascade='all, delete-orphan')

class SavedNews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(500), nullable=True)
    url = db.Column(db.String(500), nullable=True)
    original_text = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(20), nullable=False)
    num_sentences = db.Column(db.Integer, default=5)
    translated_summary = db.Column(db.Text, nullable=True)
    translated_title = db.Column(db.String(500), nullable=True)
    translated_to = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'summary': self.summary,
            'language': self.language,
            'num_sentences': self.num_sentences,
            'translated_summary': self.translated_summary,
            'translated_title': self.translated_title,
            'translated_to': self.translated_to,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=50)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=80)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=50)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=80)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

class SaveNewsForm(FlaskForm):
    title = StringField(validators=[Length(max=500)], render_kw={"placeholder": "Title"})
    url = StringField(validators=[Length(max=500)], render_kw={"placeholder": "URL"})
    summary = TextAreaField(validators=[DataRequired()], render_kw={"placeholder": "Summary"})
    language = StringField(validators=[DataRequired()], render_kw={"placeholder": "Language"})
    num_sentences = StringField(render_kw={"placeholder": "Number of sentences"})
    translated_summary = TextAreaField(render_kw={"placeholder": "Translated Summary"})
    translated_title = StringField(render_kw={"placeholder": "Translated Title"})
    translated_to = StringField(render_kw={"placeholder": "Translated to"})
    submit = SubmitField('Save News')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form, error="Invalid username or password. Please try again.")

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    saved_news = SavedNews.query.filter_by(user_id=current_user.id).order_by(SavedNews.created_at.desc()).all()
    return render_template('dashboard.html', username=current_user.username, saved_news=saved_news)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        
        except Exception as e:
            db.session.rollback()
            return render_template('register.html', form=form, error="Registration failed. Please try again.")

    return render_template('register.html', form=form)

@app.route('/save-news', methods=['POST'])
@login_required
def save_news():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        saved_news = SavedNews(
            user_id=current_user.id,
            title=data.get('title'),
            url=data.get('url'),
            original_text=data.get('original_text'),
            summary=data.get('summary'),
            language=data.get('language', 'english'),
            num_sentences=int(data.get('num_sentences', 5)),
            translated_summary=data.get('translated_summary'),
            translated_title=data.get('translated_title'),
            translated_to=data.get('translated_to')
        )
        
        db.session.add(saved_news)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'News saved successfully!',
            'news_id': saved_news.id
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving news: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/delete-news/<int:news_id>', methods=['DELETE'])
@login_required
def delete_news(news_id):
    try:
        news = SavedNews.query.get_or_404(news_id)
        
        if news.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        db.session.delete(news)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'News deleted successfully!'})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting news: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/get-saved-news', methods=['GET'])
@login_required
def get_saved_news():
    try:
        saved_news = SavedNews.query.filter_by(user_id=current_user.id).order_by(SavedNews.created_at.desc()).all()
        return jsonify({
            'success': True,
            'news': [news.to_dict() for news in saved_news]
        })
    except Exception as e:
        logger.error(f"Error fetching saved news: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.cli.command('init-db')
def init_db():
    with app.app_context():
        db.create_all()
        print("Database initialized.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)