import pickle

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import pandas as pd
import os
import glob
from collections import defaultdict
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm
import heapq

class EnglishSummarizer:
    def __init__(self, model_path='english_model.pkl'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0' 
        }
        self.ml_model = None
        self.scaler = None
        self.vectorizer = None
        
        if os.path.exists(model_path):
            self.load_model(model_path)
    
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
    
    def load_model(self, filepath='english_model.pkl'):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data.get('vectorizer')
                self.ml_model = data.get('ml_model')
                self.scaler = data.get('scaler')
                self.trained_data = data.get('trained_data')
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _extract_sentence_features(self, sentences):
        """
        Extract features for each sentence
        """
        features = []
        stop_words = set(stopwords.words('english'))
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            words = word_tokenize(sent_lower)
            words_clean = [w for w in words if w.isalpha() and w not in stop_words]
            
            # Position features
            position_score = 1.0 if i < 3 else (0.8 if i > len(sentences) - 2 else 0.5)
            
            # Length features
            word_count = len(words)
            char_count = len(sent)
            
            # Content features
            title_words = set()
            # Estimate title as first few words of first sentence
            if i == 0:
                title_words = set(words_clean[:5])
            
            title_overlap = len(set(words_clean) & title_words) / max(1, len(title_words))
            
            # Lexical features
            unique_ratio = len(set(words_clean)) / max(1, len(words_clean))
            avg_word_length = np.mean([len(w) for w in words_clean]) if words_clean else 0
            
            # NER-like features (capitalized words)
            capitalized_ratio = sum(1 for w in words if w[0].isupper()) / max(1, len(words))
            
            features.append([
                position_score,
                min(word_count / 50, 1.0),  # Normalized length
                min(char_count / 300, 1.0),  # Normalized char length
                title_overlap,
                unique_ratio,
                min(avg_word_length / 10, 1.0),
                capitalized_ratio,
                len(words_clean) / max(1, len(words)),  # Content word ratio
                i / max(1, len(sentences))  # Relative position
            ])
        
        return features
    
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

        from trainer import trainer
        trainer = trainer()

        if self.ml_model is not None and self.scaler is not None:
            try:
                features = self._extract_sentence_features(sentences)
                X_scaled = self.scaler.transform(features)
                scores = self.ml_model.predict(X_scaled)
                
                # Select top sentences
                top_indices = heapq.nlargest(num_sentences, range(len(scores)), scores.__getitem__)
                top_indices.sort()
                
                return ' '.join([sentences[i] for i in top_indices])
            except Exception as e:
                print(f"ML prediction failed, falling back to TF-IDF: {e}") 

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
        from trainer import trainer
        trainer = trainer()

        if os.path.isdir(dataset_path):
            df = trainer.load_multiple_csvs(dataset_path)
        else:
            df = trainer.load_csv_dataset(dataset_path)
        
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
        from trainer import trainer
        trainer = trainer()

        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                if trainer.load_model(path):
                    model_loaded = True
                    break
        if not model_loaded:
            trainer.train_from_csv(
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



    
        
