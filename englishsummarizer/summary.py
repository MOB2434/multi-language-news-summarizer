import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import heapq
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import pandas as pd
import os
import glob
from collections import defaultdict
import pickle
from rouge_score import rouge_scorer
from tqdm import tqdm

class EnglishSummarizer:
    def __init__(self, model_path='english_model.pkl'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0' 
        }
    
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
                soup.find('meta', {'property': 'og:title'}),
                soup.find('meta', {'name': 'twitter:title'})
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
                possible_text_cols = ['article','description']
                for col in possible_text_cols:
                    if col in df.columns:
                        text_column = col
                        print(f"Using '{text_column}' as text column")
                        break
            
            if summary_column not in df.columns:
                print(f"Warning: '{summary_column}' column not found.")
                possible_summary_cols = [ 'highlights', 'title']
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
        
        self.save_model('english_model.pkl')

        return True
     
    def save_model (self, filepath='english_model.pkl'):

        if self.vectorizer is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'trained_data': self.trained_data
                }, f)
            print(f"Model saved to {filepath}")
            return True
        else:
            print("No trained model to save")
            return False
    
    def load_model(self, filepath='english_model.pkl'):

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.trained_data = data['trained_data']
            print(f"Model loaded from {filepath}")
            print(f"  • Trained on {self.trained_data['num_articles']} articles")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def summarize(self, text, num_sentences=5):
        if not text or len(text.split()) < 10:
            return text
    
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
    
        stop_words = set(stopwords.words('english'))
        cleaned_sentences = []
        original_sentences = []
    
        for sent in sentences:
            sent_clean = re.sub(r'[^a-zA-Z\s]', '', sent)
            words = word_tokenize(sent_clean.lower())
            words = [word for word in words if word not in stop_words and len(word) > 2]
            if words:  
                cleaned_sentences.append(' '.join(words))
                original_sentences.append(sent)
    
        if len(original_sentences) <= num_sentences:
            return ' '.join(original_sentences)
    
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_sentences)
    
        position_scores = []
        for i in range(len(original_sentences)):
            if i < 3:
                pos_score = 1.0
            elif i > len(original_sentences) - 2:
                pos_score = 0.8
            else:
                pos_score = 0.5
            position_scores.append(pos_score)
    
        similarity_matrix = cosine_similarity(tfidf_matrix)
    
        sentence_scores = []
        for i in range(len(original_sentences)):
            doc_similarity = similarity_matrix[i].sum()
        
            word_count = len(original_sentences[i].split())
            if word_count < 5:
                length_score = 0.3
            elif word_count > 50:
                length_score = 0.5
            else:
                length_score = 1.0
        
            total_score = (doc_similarity * 0.5 + 
                      position_scores[i] * 0.3 + 
                      length_score * 0.2)
            sentence_scores.append(total_score)
    
        selected_indices = []
        for _ in range(min(num_sentences, len(original_sentences))):
            if not selected_indices:
                idx = np.argmax(sentence_scores)
            else:
                remaining_scores = sentence_scores.copy()
                for selected in selected_indices:
                    for i in range(len(remaining_scores)):
                        if i not in selected_indices:
                            penalty = similarity_matrix[i][selected] * 0.5
                            remaining_scores[i] -= penalty
            
                for idx in selected_indices:
                    remaining_scores[idx] = -np.inf
            
                idx = np.argmax(remaining_scores)
        
            selected_indices.append(idx)
    
        selected_indices.sort()
        summary = [original_sentences[i] for i in selected_indices]
        return ' '.join(summary)
    
    def summarize_url(self, url, num_sentences=5):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        title, text = self.fetch_article(url)
    
        if text and not text.startswith("Error"):
            summary = self.summarize(text, num_sentences)
            
            print(f"   Original: ~{len(text.split())} words")
            print(f"   Summary: {len(summary.split())} words")
            print(f"Title: {title}\n  ")
            print(f"\n   {summary}\n")
            
        else:
            print(f"{text}\n")
            return text

    def calculate_rouge_scores(self, test_data, num_sentences=3):
        print("calculating ROUGE scores")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        all_scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        successful = 0
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Calculating ROUGE scores"):
            try:
                article = row['cleaned_text']
                reference_summary = row['cleaned_summary']
                
                if pd.isna(article) or pd.isna(reference_summary) or len(article.split()) < 20:
                    continue
                
                generated_summary = self.summarize(article, num_sentences)
                
                scores = scorer.score(reference_summary, generated_summary)
                
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    all_scores[metric]['precision'].append(scores[metric].precision)
                    all_scores[metric]['recall'].append(scores[metric].recall)
                    all_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
                
                successful += 1
                
                if idx == 0:
                    print(f"Reference: {reference_summary[:200]}...")
                    print(f"Generated: {generated_summary[:200]}...")
                    print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.4f}")
                    print(f"ROUGE-2 F1: {scores['rouge2'].fmeasure:.4f}")
                    print(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        if successful == 0:
            print("No valid test samples found")
            return None
        
        avg_scores = {}
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            avg_scores[metric] = {
                'precision': np.mean(all_scores[metric]['precision']),
                'recall': np.mean(all_scores[metric]['recall']),
                'fmeasure': np.mean(all_scores[metric]['fmeasure'])
            }
    
        print(f"Tested on {successful} samples")
        print("\n{:<10} {:<12} {:<12} {:<12}".format("Metric", "Precision", "Recall", "F1-Score"))
        print("-"*48)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            print("{:<10} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                metric.upper(),
                avg_scores[metric]['precision'],
                avg_scores[metric]['recall'],
                avg_scores[metric]['fmeasure']
            ))
        
        return avg_scores

    def evaluate_on_dataset(self, dataset_path, test_split=0.2, num_sentences=3):
        print("Model Evaluation:")
        
        if os.path.isdir(dataset_path):
            df = self.load_multiple_csvs(dataset_path)
        else:
            df = self.load_csv_dataset(dataset_path)
        
        if df is None or df.empty:
            print("Failed to load dataset")
            return None
        
        if 'cleaned_summary' not in df.columns:
            print("No summary column found. Cannot calculate ROUGE scores.")
            return None
        
        df = df.dropna(subset=['cleaned_text', 'cleaned_summary'])
        df = df[df['cleaned_text'].str.len() > 50]  
        df = df[df['cleaned_summary'].str.len() > 10]  
        
        print(f"Valid samples after cleaning: {len(df)}")
        
        if len(df) == 0:
            print("No valid samples for evaluation")
            return None
        
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_split, random_state=42)
        
        print(f"\nUsing {len(test_df)} samples for evaluation")
        
        rouge_scores = self.calculate_rouge_scores(test_df, num_sentences)
        
        return rouge_scores

def main():
    summarizer = EnglishSummarizer()

    dataset_path = "/home/banshika/multi-language-news-summarizer/englishsummarizer/datasets" 
    model_path = 'english_model.pkl'

    while True:
        print("Summarize URL")
        
        possible_paths = [
            model_path,
            'models/english_model.pkl',
            'english_model.pkl',
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



    
        
