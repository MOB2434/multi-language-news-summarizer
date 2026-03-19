from email.mime import text
import requests
from bs4 import BeautifulSoup
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
from collections import defaultdict, Counter
import pickle
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
from tqdm import tqdm
import networkx as nx
import math
from sklearn.metrics.pairwise import cosine_similarity

class NepaliSummarizer:
    def __init__(self, model_path='nepali_model.pkl'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0' 
        }
        self.trained_data = None
        self.textrank_alpha = 0.85  
        self.mmr_lambda = 0.5 
        self.use_position_bias = True
        self.use_length_filter = True
        self.min_sentence_words = 8  
        self.max_sentence_words = 40 
        self.important_words = set()
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
                'उनको','उन','उप','उहाँलाई','एउटै','एकदम','एक','औं','कतै','कम से कम','कसैले'
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
        
        self.trained_data = {
            'num_articles': len(articles),
            'avg_article_length': np.mean([len(art.split()) for art in articles]),
            'columns_used': {
                'text': text_column,
                'summary': summary_column if summary_column in df.columns else None
            }
        }
        
        print(f"\nTraining data processed successfully!")
        print(f"  Articles: {self.trained_data['num_articles']}")
        print(f"  Average article length: {self.trained_data['avg_article_length']:.0f} words")
        
        self.save_model('nepali_model.pkl')
    
        return True
     
    def save_model (self, filepath='nepali_model.pkl'):
        model_data = {
            'trained_data': self.trained_data,
            'nepali_stopwords': self.nepali_stopwords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='nepali_model.pkl'):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                self.trained_data = model_data.get('trained_data', None)
                self.nepali_stopwords = model_data.get('nepali_stopwords', self.nepali_stopwords)
                print(f"Model loaded from {filepath}")
                if self.trained_data:
                    print(f"  Trained on {self.trained_data.get('num_articles', 0)} articles")
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_text(self, text):
        if not text or pd.isna(text):
            return ""
    
        text = str(text)  
        text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_sentences(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[।\?!]+\s*', text)
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            words = sent.split()
            if 3 <= len(words) <= 100 and len(sent) > 5: 
                cleaned_sentences.append(sent)
        
        if not cleaned_sentences:
            cleaned_sentences = [s for s in sentences if len(s.strip()) > 0]
        
        return cleaned_sentences
    
    def get_word_occurrence_vector(self, sentences):
        if not sentences:
            return np.array([])
    
        word_freq = Counter()
        sentence_words = []
        for sent in sentences:
            words = sent.split()
            sentence_words.append(words)
            word_freq.update(words)
    
        total_sentences = len(sentences)
        min_df = max(2, int(total_sentences * 0.05))
        max_df = max(min_df + 1, int(total_sentences * 0.8))
    
        vocab = [word for word, freq in word_freq.items() 
            if min_df <= freq <= max_df and len(word) > 1]  
    
        vocab = vocab[:5000]  
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
        idf = {}
        for word in vocab:
            doc_count = sum(1 for words in sentence_words if word in words)
            idf[word] = math.log((total_sentences + 1) / (doc_count + 1)) + 1
    
        vectors = []
        for words in sentence_words:
            vector = np.zeros(len(vocab))
            word_counts = Counter(words)
        
            for word, count in word_counts.items():
                if word in word_to_idx:
                    tf = 1 + math.log(count)
                    vector[word_to_idx[word]] = tf * idf[word]
        
            vectors.append(vector)
    
        return np.array(vectors)

    def build_similarity_matrix(self, sentences):
        vectors = self.get_word_occurrence_vector(sentences)
        
        if vectors.size == 0:
            return np.zeros((len(sentences), len(sentences)))

        cosine_sim = cosine_similarity(vectors)
        
        jaccard_sim = np.zeros((len(sentences), len(sentences)))
        sentence_sets = [set(s.split()) for s in sentences]

        for i in range(len(sentences)):
            words_i = sentence_sets[i]
            for j in range(len(sentences)):
                words_j = sentence_sets[j]
                if words_i and words_j:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    jaccard_sim[i][j] = intersection / union if union > 0 else 0
        
        similarity_matrix = 0.6 * cosine_sim + 0.4 * jaccard_sim
        
        non_zero = similarity_matrix[similarity_matrix > 0]
        threshold = np.mean(non_zero) * 0.5 if len(non_zero) > 0 else 0
        similarity_matrix[similarity_matrix < threshold] = 0
        
        return similarity_matrix

    def textrank(self, sentences, damping=0.85, max_iter=200, tol=1e-4):
        if len(sentences) <= 1:
            return [(0, 1.0)], {0: 1.0}
    
        sim_matrix = self.build_similarity_matrix(sentences)
    
        if np.sum(sim_matrix) == 0:
            uniform_score = 1.0/len(sentences)
            return [(i, uniform_score) for i in range(len(sentences))], \
                {i: uniform_score for i in range(len(sentences))}
    
        n = len(sentences)
        scores = np.ones(n) / n
        dangling_weights = np.ones(n) / n
    
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = sim_matrix / row_sums
    
        for _ in range(max_iter):
            prev_scores = scores.copy()
        
            scores = (damping * np.dot(transition_matrix.T, scores) + (1 - damping) * dangling_weights)
        
            if np.linalg.norm(scores - prev_scores) < tol:
                break
    
        scores_dict = {i: scores[i] for i in range(n)}
        ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    
        return ranked, scores_dict

    def calculate_position_score(self, idx, total_sentences):
        if not self.use_position_bias:
            return 1.0
    
        if total_sentences <= 5:
            return 1.0 - 0.15 * idx
    
        position = idx / total_sentences
    
        if position < 0.2:
            return 1.0 - position * 2  
        elif position < 0.4:
            return 0.6 - (position - 0.2) * 1.5  
        else:
            return max(0.2, 0.3 - (position - 0.4) * 0.25) 

    def calculate_length_score(self, sentence):
        if not self.use_length_filter:
            return 1.0
    
        words = sentence.split()
        word_count = len(words)
    
        if 10 <= word_count <= 20:  
            return 1.0
        elif 7 <= word_count < 10:
            return 0.85
        elif 20 < word_count <= 25:
            return 0.85
        elif 4 <= word_count < 7:
            return 0.6
        elif 25 < word_count <= 30:
            return 0.6
        elif 30 < word_count <= 40:
            return 0.4
        else:
            return 0.2

    def calculate_keyword_score(self, sentence):
        words = sentence.split()
        if not words:
            return 0
        
        score = 0
        total_weight = 0
        
        for word in words:
            word_weight = 1
            
            if word.istitle():
                word_weight += 1.5
            
            if word in self.important_words:
                word_weight += 1.0
            
            if word not in self.nepali_stopwords and len(word) > 3:
                word_weight += 0.5
            
            score += word_weight
            total_weight += 1
        
        base_score = score / max(total_weight, 1)
        
        rare_word_count = sum(1 for w in words if len(w) > 5 and w not in self.nepali_stopwords)
        if rare_word_count >= 2:
            base_score *= 1.2
        
        return min(base_score, 2.5)

    def combine_scores(self, textrank_scores, sentences):
        total_sentences = len(sentences)
        
        textrank_values = np.array([textrank_scores[i] for i in range(total_sentences)])
        if textrank_values.max() > textrank_values.min():
            textrank_norm = (textrank_values - textrank_values.min()) / (textrank_values.max() - textrank_values.min())
        else:
            textrank_norm = np.ones(total_sentences)
        
        textrank_norm = np.power(textrank_norm, 0.8)
        
        position_scores = np.array([self.calculate_position_score(i, total_sentences) for i in range(total_sentences)])
        
        length_scores = np.array([self.calculate_length_score(sent) for sent in sentences])
        
        keyword_scores = np.array([self.calculate_keyword_score(sent) for sent in sentences])
        
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
        
        if total_sentences < 10:
            weights = {
                'textrank': 0.25,
                'position': 0.40,  
                'length': 0.10,
                'keywords': 0.25
            }
        elif total_sentences < 20:
            if avg_sentence_length > 20:
                weights = {'textrank': 0.40, 'position': 0.25, 'length': 0.10, 'keywords': 0.25}
            else:
                weights = {'textrank': 0.35, 'position': 0.30, 'length': 0.15, 'keywords': 0.20}
        else:
            weights = {'textrank': 0.45, 'position': 0.15, 'length': 0.15, 'keywords': 0.25}
        
        final_scores = (weights['textrank'] * textrank_norm +
                    weights['position'] * position_scores +
                    weights['length'] * length_scores +
                    weights['keywords'] * keyword_scores)
        
        final_scores += np.arange(total_sentences) * 1e-9
        if final_scores.max() > final_scores.min():
            final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min())
            
        return final_scores

    def mmr_selection(self, sentences, scores, num_sentences=5, lambda_param=None):
        if lambda_param is None:
            lambda_param = self.mmr_lambda
        
        if len(sentences) <= num_sentences:
            return list(range(len(sentences)))
        
        if len(sentences) > 30:
            lambda_param = max(0.5, lambda_param - 0.1)  
        
        vectors = self.get_word_occurrence_vector(sentences)
        
        if vectors.size == 0 or len(vectors) == 0:
            return sorted(range(len(sentences)), key=lambda x: scores[x], reverse=True)[:num_sentences]
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        sim_matrix = np.dot(vectors, vectors.T)
        
        selected = []
        candidates = list(range(len(sentences)))
        
        candidates.sort(key=lambda x: scores[x], reverse=True)
        
        while len(selected) < min(num_sentences, len(sentences)):
            if not candidates:
                break
            
            if not selected:
                selected.append(candidates[0])
                candidates.pop(0)
            else:
                mmr_scores = []
                for idx in candidates:
                    relevance = scores[idx]
                    
                    if selected:
                        max_sim = np.max([sim_matrix[idx][s] for s in selected])
                    else:
                        max_sim = 0
                    
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr)
                
                best_idx = candidates[np.argmax(mmr_scores)]
                selected.append(best_idx)
                candidates.remove(best_idx)
        
        return sorted(selected)

    def summarize(self, text, num_sentences=5):
        if not text or len(text.split()) < 20:
            return text
    
        sentences = self.extract_sentences(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
    
        if len(sentences) > 50:
            num_sentences = min(num_sentences + 1, 8)

        vectors = self.get_word_occurrence_vector(sentences)
        sim_matrix = self.build_similarity_matrix(sentences)  
        ranked_sentences, scores_dict = self.textrank(sentences, damping=self.textrank_alpha)
        textrank_scores = [scores_dict.get(i, 0) for i in range(len(sentences))]
        combined_scores = self.combine_scores(textrank_scores, sentences)
        selected_indices = self.mmr_selection(sentences, combined_scores, num_sentences)
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices[:num_sentences]]
    
        first_sentence = sentences[0] if sentences else ""
        if (first_sentence and 
            self.use_position_bias and 
            first_sentence not in summary_sentences):
        
            first_score = combined_scores[0]
            if first_score > np.percentile(combined_scores, 70):  
                lowest_idx = min(selected_indices, key=lambda x: combined_scores[x])
                if combined_scores[0] > combined_scores[lowest_idx]:
                    selected_indices.remove(lowest_idx)
                    selected_indices.append(0)
                    selected_indices.sort()
                    summary_sentences = [sentences[i] for i in selected_indices[:num_sentences]]
    
        summary = "। ".join(summary_sentences)
    
        if summary and not summary.endswith('।'):
            summary += '।'
    
        if summary and summary[0].isalpha():
            summary = summary[0].upper() + summary[1:]
    
        return summary

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
        
    def calculate_rouge_scores(self, reference_summaries, generated_summaries, metrics=['rouge1', 'rouge2', 'rougeL']):
        if len(reference_summaries) != len(generated_summaries):
            print(f"Warning: Number of references ({len(reference_summaries)}) and generations ({len(generated_summaries)}) don't match")
            return None
        
        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=False)
        
        all_scores = {metric: {'precision': [], 'recall': [], 'fmeasure': []} for metric in metrics}
        
        print("\nCalculating ROUGE scores...")
        for ref, gen in tqdm(zip(reference_summaries, generated_summaries), total=len(reference_summaries)):
            if isinstance(ref, np.ndarray):
                ref = str(ref)
            if isinstance(gen, np.ndarray):
                gen = str(gen)
            if pd.isna(ref):
                ref = ""
            if pd.isna(gen):
                gen = ""

            ref = str(ref) if not isinstance(ref, str) else ref
            gen = str(gen) if not isinstance(gen, str) else gen

            if not ref.strip() and not gen.strip():
                continue

            try:
                scores = scorer.score(ref, gen)
            
                for metric in metrics:
                    all_scores[metric]['precision'].append(scores[metric].precision)
                    all_scores[metric]['recall'].append(scores[metric].recall)
                    all_scores[metric]['fmeasure'].append(scores[metric].fmeasure)

            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
    
        avg_scores = {}
        for metric in metrics:
            avg_scores[metric] = {
                'precision': np.mean(all_scores[metric]['precision']),
                'recall': np.mean(all_scores[metric]['recall']),
                'fmeasure': np.mean(all_scores[metric]['fmeasure'])
            }
        
        return avg_scores
    
    def evaluate_on_dataset(self, dataset_path, text_column='cleaned_text', summary_column='cleaned_summary', num_sentences=5, sample_size=None):
        if os.path.isdir(dataset_path):
            df = self.load_multiple_csvs(dataset_path)
        else:
            df = self.load_csv_dataset(dataset_path, text_column, summary_column)
        
        if df is None or df.empty:
            print("No data available for evaluation")
            return None
        
        if 'cleaned_text' not in df.columns or 'cleaned_summary' not in df.columns:
            print("Required columns not found in dataset")
            return None

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} articles for evaluation")
        
        articles = df['cleaned_text'].tolist()
        reference_summaries = df['cleaned_summary'].tolist()
        
        print(f"\nGenerating summaries for {len(articles)} articles...")
        generated_summaries = []
        
        for article in tqdm(articles):
            summary = self.summarize(article, num_sentences=num_sentences)
            generated_summaries.append(summary)
        
        rouge_scores = self.calculate_rouge_scores(reference_summaries, generated_summaries)
        
        results = {
            'rouge_scores': rouge_scores,
            'generated_summaries': generated_summaries,
            'reference_summaries': reference_summaries,
            'num_samples': len(articles),
            'num_sentences': num_sentences
        }
        
        return results
       
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
                text_column='text',
                summary_column='summary'
            )
        
        url = input("Enter URL (or type 'quit' to exit): ").strip()
        if url.lower() in ('quit', 'exit'):
            print("Exiting.")
            break
        summarizer.summarize_url(url)
            
if __name__ == "__main__":
    main()