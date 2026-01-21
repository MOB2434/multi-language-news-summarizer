from pydoc import text
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq
import nltk
import string
import sys 
sys.path.append('..')
import re
import pandas as pd
import os
import glob
from collections import defaultdict
import pickle
from nltk.corpus import stopwords

class NepaliSummarizer:
    def __init__(self, model_path='nepali_model.pkl'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0' 
        }
        self.vectorizer = None
        try:
            self.nepali_stopwords = set(stopwords.words('nepali'))
        except:
            self.nepali_stopwords = set([
                'र', 'को', 'का', 'मा', 'हो', 'गर्न', 'भने', 'लाई', 'ले', 'बाट',
                'तथापि', 'यस', 'यो', 'भन्ने', 'गरे', 'छ', 'छन्', 'भएको', 'त्यस',
                'पनि', 'तर', 'अनि', 'कि', 'जुन', 'जस', 'जस्तो', 'छैन', 'थियो', 'थिए',
                'गरेको', 'गर्दै', 'गरिरहेको', 'भए', 'भएको', 'गर्नु', 'हुन्छ', 'हुने', 'हुनेछ',
                'गर्नुपर्ने', 'गर्नुहोस्', 'भएकोछ', 'गर्नेलाई', 'भएकोछ', 'गर्नेलाई','भएर','अक्सर','अगाडि','अझै','अनुसार',
                'अन्तर्गत','अन्य','अन्यत्र','अन्यथा','अब','अरू','अरूलाई','अर्को','अर्थात','अर्थात्','अलग','आए','आजको',
                'आफ्ना','आफ्नासँग','आफ्नो','आफ्नै','आफूले','आफूलाई','आफैलाई','आफू','आयो','उदाहरण','उनले',
                'उनको','उन','उप','उहाँलाई','एउटै','एकदम','एक','औं','कतै','कम से कम','कसैले',
            ])
    
    def fetch_article(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')

            for elem in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'figure', 'blockquote', 'ul', 'ol', 'table', 'form']):
                elem.decompose()

            for element in soup.find_all(['sup', 'span', 'a'], class_=re.compile('reference|citation')):
                element.decompose()

            for byline in soup.find_all(class_=re.compile('byline|author|timestamp|dateline|published')):
                byline.decompose()  
            for time_tag in soup.find_all(['time',]):
                time_tag.decompose()
            
            title = ""
            
            title_selectors = [
                soup.find('h1'),
                soup.find(class_=re.compile('headline|title|article-title|story-title')),
                soup.find('title'),
            ]
            
            for selector in title_selectors:
                if selector:
                    if hasattr(selector, 'get_text'):
                        title = selector.get_text(strip=True)
                    elif selector.get('content'):
                        title = selector.get('content', '').strip()
                    
                    if title and len(title) > 10: 
                        break
            
            if not title or len(title) < 5:
                title = url.split('/')[-1].replace('-', ' ').title()
                if len(title) < 5:
                    title = "Article Summary"

            article = soup.find('article') or soup.find(class_=re.compile('article|post|story|content|main|body|section'))
            
            if article:
                text = article.get_text()
            else:
                text = soup.find('body').get_text()

            paragraphs = []
            for p in article.find_all(['p', 'h2', 'h3'], recursive=True):
                text = p.get_text(' ', strip=True)
                if len(text.split()) > 10 and not any(x in text.lower() for x in ['sign up', 'subscribe', 'follow us']):
                    paragraphs.append(text)
            full_text = ' '.join(paragraphs)
            
            text = re.sub(r'\s+', ' ', full_text)
            text = re.sub(r'\[.*?\]|\(.*?\)', '', full_text)
            text = re.sub(r'[a-zA-Z0-9]', '', full_text)
            
            return title.strip(), text.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def load_csv_dataset(self, csv_path, text_column='text', summary_column='summary'):
        
        print(f"Loading dataset from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded: {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            
            if text_column not in df.columns:
                print(f"Warning: '{text_column}' column not found. Available columns: {df.columns.tolist()}")
                possible_text_cols = ['paras', 'text']
                for col in possible_text_cols:
                    if col in df.columns:
                        text_column = col
                        print(f"Using '{text_column}' as text column")
                        break
            
            if summary_column not in df.columns:
                print(f"Warning: '{summary_column}' column not found.")
                possible_summary_cols = ['title', 'headings']
                for col in possible_summary_cols:
                    if col in df.columns:
                        summary_column = col
                        print(f"Using '{summary_column}' as summary column")
                        break
            
            df['cleaned_text'] = df[text_column].apply(self.clean_dataset_text)
            
            if summary_column in df.columns:
                df['cleaned_summary'] = df[summary_column].apply(self.clean_dataset_text)
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def load_multiple_csvs(self, folder_path):
        
        all_data = []
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return None
        
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {folder_path}")
            return None
        
        print(f"Found {len(csv_files)} CSV files:")
        
        for csv_file in csv_files:
            df = self.load_csv_dataset(csv_file)
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"  ✓ {os.path.basename(csv_file)}: {len(df)} rows")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal combined dataset: {len(combined_df)} rows")
            return combined_df
        
        return None
    
    def clean_dataset_text(self, text):

        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]|\(\d+\)', '', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
    
    def train_from_csv(self, csv_path_or_folder, text_column='text', summary_column='summary'):
       
        if os.path.isdir(csv_path_or_folder):
            df = self.load_multiple_csvs(csv_path_or_folder)
        else:
            df = self.load_csv_dataset(csv_path_or_folder, text_column, summary_column)
        
        if df is None or df.empty:
            print("No data available for training")
            return False
        
        articles = df['cleaned_text'].tolist()
        
        all_text = ' '.join(articles)
        
        print("\nTraining TF-IDF vectorizer")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000, 
            ngram_range=(1, 2) 
        )
        
        self.vectorizer.fit(articles)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        self.trained_data = {
            'num_articles': len(articles),
            'avg_article_length': np.mean([len(art.split()) for art in articles]),
            'vocab_size': len(self.vectorizer.vocabulary_),
            'columns_used': {
                'text': text_column,
                'summary': summary_column if summary_column in df.columns else None
            }
        }
        
        print("\nTraining completed successfully!")
        print(f"  • Articles trained on: {self.trained_data['num_articles']}")
        print(f"  • Average article length: {self.trained_data['avg_article_length']:.0f} words")
        print(f"  • Vocabulary size: {self.trained_data['vocab_size']}")
        
        self.save_model('nepali_model.pkl')

        return True
     
    def save_model (self, filepath='nepali_model.pkl'):

        if self.vectorizer is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'trained_data': self.trained_data,
                    'stopwords': self.nepali_stopwords
                }, f)
            print(f"Model saved to {filepath}")
            return True
        else:
            print("No trained model to save")
            return False
    
    def load_model(self, filepath='nepali_model.pkl'):

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.trained_data = data['trained_data']
                if 'stopwords' in data:
                    self.nepali_stopwords = data['stopwords']
            print(f"Model loaded from {filepath}")
            print(f"  • Trained on {self.trained_data['num_articles']} articles")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def word_tokenize(self, text):
        words = re.findall(r'[\u0900-\u097F]+|[^\s\w]|\w+', text)
        return words
        
    def sent_tokenize(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        pattern = r'(?<![डपशकखगजच]\.)(?<![डपशकखगजच]॰)(?<!\d\.)(?<=[।।।\?\!])\s+'
        sentences = re.split(pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def summarize(self, text, num_sentences=5):
        
        if not text or len(text.split()) < 10:
            return text
        
        sentences = self.sent_tokenize(text)
        clean_sentences = []
    
        for sentence in sentences:
            words = self.word_tokenize(sentence)
            clean_words = []
            for word in words:
                has_letters = any(char.isalpha() or char.isdigit() for char in word)
                if has_letters and word not in self.nepali_stopwords:
                    clean_words.append(word)
            clean_sentences.append(" ".join(clean_words))
           
        try:
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(clean_sentences)
            cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
            sentence_scores = cosine_similarities.sum(axis=1)
            top_sentence_indices = heapq.nlargest(num_sentences, range(len(sentence_scores)), key=sentence_scores.take)
            top_sentence_indices.sort()
            summary = [sentences[i] for i in sorted(top_sentence_indices)]
            return " ".join(summary)
        
        except Exception as e:
            print(f"Error using trained model: {e}")
            return self.summarize(text, num_sentences)
    
    def summarize_url(self, url, num_sentences=5):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        title, text = self.fetch_article(url)
    
        if text and not text.startswith("Error"):
            if self.vectorizer is not None:
                summary = self.summarize(text, num_sentences)
                model_type = "Nepali Model"
                
            print(f"   Original: ~{len(text.split())} words")
            print(f"   Summary: {len(summary.split())} words")
            print(f"Title: {title}\n  ")
            print(f"\n   {summary}\n")

            try:
                filename = f"nepalisummary.txt"
                foldername = "summaries"
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                filepath = os.path.join(foldername, filename)
                content = []
                content.append(f" {summary}\n")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(''.join(content))
                return filepath
            except Exception as e:
                print(f"Error saving summary: {e}")
                return summary
        else:
            print(f"{text}\n")
            return text 

def main():
    summarizer = NepaliSummarizer()

    dataset_path = "/home/banshika/multi-language-news-summarizer/nepalisummarizer/datasets" 
    model_path = 'nepali_model.pkl'

    while True:
        print("Summarize URL")
                
        possible_paths = [
            model_path,
            'models/nepali_model.pkl',
            'nepali_model.pkl',
        ]
        
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                if summarizer.load_model(path):
                    model_loaded = True
                    break
                
        if not model_loaded:
            summarizer.train_from_csv(
                csv_path_or_folder=dataset_path,
                text_column='article',
                summary_column='summary'
            )
        
        url = input("Enter URL (or type 'quit' to exit): ").strip()
        if url.lower() in ('quit', 'exit'):
            print("Exiting.")
            break
        summarizer.summarize_url(url)
            
if __name__ == "__main__":
    main()
    