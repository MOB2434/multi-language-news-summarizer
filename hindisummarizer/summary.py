from email.mime import text
from turtle import position
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

class HindiSummarizer:
    def __init__(self, model_path='hindi_model.pkl'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0' 
        }
        self.trained_data = None
        self.textrank_alpha = 0.95
        self.mmr_lambda = 0.9 
        self.use_position_bias = True
        self.use_length_filter = True
        self.min_sentence_words = 5  
        self.max_sentence_words = 50 
        self.important_words = set()
        try:
            self.hindi_stopwords = set(stopwords.words('hindi'))
        except:
            self.hindi_stopwords = set([
                'के','का','एक','में','की','है','यह','और','से','हैं','को','पर','इस',
                'होता','कि','जो','कर','मे','गया','करने','किया','लिये','अपने','ने','बनी',
                'नहीं','तो','ही','या','एवं','दिया','हो','इसका','था','द्वारा','हुआ','तक',
                'साथ','करना','वाले','बाद','लिए','आप','कुछ','सकते','किसी','ये','इसके',
                'सबसे','इसमें','थे','दो','होने','वह','वे','करते','बहुत','कहा','वर्ग','कई',
                'करें','होती','अपनी','उनके','थी','यदि','हुई','जा','ना','इसे','कहते','जब','होते','कोई',
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
    
    def preprocess_text(self, text):
        if not text or pd.isna(text):
            return ""
    
        text = str(text)  
        text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_sentences(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[।!?\.]+', text)
        
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 5:
                cleaned.append(sent)
        
        return cleaned

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

    def calculate_keyword_score(self, sentence):
        words = sentence.split()
        if not words:
            return 0
        
        score = 0
        for word in words:
            if word in self.important_words:
                score += 1.5
            elif word not in self.hindi_stopwords and len(word) > 2:
                score += 1.0
            else:
                score += 0.2
        
        base_score = score / len(words)
        return min(base_score, 2.5)

    def calculate_length_score(self, sentence):
        word_count = len(sentence.split())
        
        if hasattr(self, 'trained_patterns'):
            optimal = self.trained_patterns.get('optimal_sentence_length', 22)
            deviation = abs(word_count - optimal) / optimal
            
            if deviation < 0.2:
                return 1.0
            elif deviation < 0.4:
                return 0.7
            elif deviation < 0.6:
                return 0.4
            else:
                return 0.2
        else:
            if 10 <= word_count <= 20:
                return 1.0
            elif 7 <= word_count < 10 or 20 < word_count <= 25:
                return 0.85
            elif 4 <= word_count < 7 or 25 < word_count <= 30:
                return 0.6
            else:
                return 0.2

    def textrank(self, sentences):
        n = len(sentences)
        if n == 0:
            return [], {}
        
        importance_scores = []
        for sentence in sentences:
            importance_scores.append(self.calculate_keyword_score(sentence))
        
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                words_i = set(sentences[i].split())
                words_j = set(sentences[j].split())
                
                if not words_i or not words_j:
                    continue
                
                intersection = words_i.intersection(words_j)
                union = words_i.union(words_j)
                
                if union:
                    weighted_intersection = sum(
                        self.word_influence_dict.get(w, 1.0) for w in intersection
                    ) if hasattr(self, 'word_influence_dict') else len(intersection)
                    
                    weighted_union = sum(
                        self.word_influence_dict.get(w, 1.0) for w in union
                    ) if hasattr(self, 'word_influence_dict') else len(union)
                    
                    similarity = (weighted_intersection / weighted_union) * (importance_scores[i] + importance_scores[j]) / 2
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        damping = self.textrank_alpha
        scores = np.ones(n) / n
        
        for _ in range(200):
            new_scores = np.zeros(n)
            for i in range(n):
                incoming_sum = 0
                for j in range(n):
                    if j != i and similarity_matrix[j][i] > 0:
                        outgoing_sum = sum(similarity_matrix[j])
                        if outgoing_sum > 0:
                            incoming_sum += similarity_matrix[j][i] / outgoing_sum * scores[j]
                new_scores[i] = (1 - damping) + damping * incoming_sum
            
            if np.sum(np.abs(new_scores - scores)) < 1e-4:
                scores = new_scores
                break
            scores = new_scores
        
        combined_scores = scores * importance_scores
        
        ranked_indices = np.argsort(combined_scores)[::-1]
        ranked_sentences = [sentences[i] for i in ranked_indices]
        scores_dict = {i: combined_scores[i] for i in range(n)}
        
        return ranked_sentences, scores_dict

    def combine_scores(self, textrank_scores, sentences):
        total_sentences = len(sentences)
        
        if hasattr(self, 'optimal_weights'):
            weights = self.optimal_weights
        else:
            weights = {'textrank': 0.35, 'position': 0.25, 'length': 0.15, 'keywords': 0.25}
        
        textrank_values = np.array(textrank_scores)
        if textrank_values.max() > textrank_values.min():
            textrank_norm = (textrank_values - textrank_values.min()) / (textrank_values.max() - textrank_values.min())
        else:
            textrank_norm = np.ones(total_sentences)
        
        position_scores = np.array([self.calculate_position_score(i, total_sentences) for i in range(total_sentences)])
        length_scores = np.array([self.calculate_length_score(sent) for sent in sentences])
        keyword_scores = np.array([self.calculate_keyword_score_enhanced(sent) for sent in sentences])
        
        if keyword_scores.max() > keyword_scores.min():
            keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
        
        final_scores = (weights['textrank'] * textrank_norm +
                    weights['position'] * position_scores +
                    weights['length'] * length_scores +
                    weights['keywords'] * keyword_scores)
        
        return final_scores

    def calculate_position_score(self, idx, total_sentences):
        if not self.use_position_bias:
            return 1.0
        
        position = idx / total_sentences if total_sentences > 0 else 0
        
        if hasattr(self, 'trained_patterns'):
            bias = self.trained_patterns.get('position_bias', 0.2)
            if position < bias:
                return 1.0 - (position / bias) * 0.3
            elif position < 0.8:
                return 0.7 - (position - bias) * 0.5
            else:
                return 0.6
        else:
            if position < 0.2:
                return 1.0
            elif position < 0.4:
                return 0.8
            elif position < 0.6:
                return 0.6
            elif position < 0.8:
                return 0.7
            else:
                return 0.9

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
        
        ranked_sentences, scores_dict = self.textrank(sentences)
        textrank_scores = [scores_dict.get(i, 0) for i in range(len(sentences))]
        
        combined_scores = self.combine_scores(textrank_scores, sentences)
        
        selected_indices = self.mmr_selection(sentences, combined_scores, num_sentences)
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices[:num_sentences]]
        
        first_sentence = sentences[0] if sentences else ""
        if (first_sentence and self.use_position_bias and 
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
        from trainer import trainer
        trainer = trainer()
        
        if os.path.isdir(dataset_path):
            df = trainer.load_multiple_csvs(dataset_path)
        else:
            df = trainer.load_csv_dataset(dataset_path, text_column, summary_column)
        
        if df is None or df.empty:
            print("No data available for evaluation")
            return None
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Evaluating on {sample_size} samples")
        
        # Generate summaries
        print(f"\nGenerating summaries for {len(df)} articles...")
        generated_summaries = []
        
        for article in tqdm(df['cleaned_text']):
            summary = self.summarize(article, num_sentences)
            generated_summaries.append(summary)
        
        # Calculate scores
        rouge_scores = self.calculate_rouge_scores(df['cleaned_summary'], generated_summaries)
            
        results = {
            'rouge_scores': rouge_scores,
            'generated_summaries': generated_summaries,
            'num_sentences': num_sentences,
        }
        
        return results
    
    def load_model(self, filepath='hindi_model.pkl'):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_data = model_data.get('trained_data', None)
            self.hindi_stopwords = model_data.get('hindi_stopwords', self.hindi_stopwords)
            self.important_words = model_data.get('important_words', set())
            
            if self.trained_data:
                self.min_sentence_words = self.trained_data.get('min_sentence_words', self.min_sentence_words)
                self.max_sentence_words = self.trained_data.get('max_sentence_words', self.max_sentence_words)
            
            print(f"Model loaded from {filepath}")
            if self.trained_data:
                print(f"  Trained on {self.trained_data.get('num_articles', 0)} articles")
                print(f"  Learned {self.trained_data.get('important_words_count', 0)} important words")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
def main():
    summarizer = HindiSummarizer()

    dataset_path = "/home/banshika/multi-language-news-summarizer/hindisummarizer/datasets" 
    model_path = 'hindi_model.pkl'

    while True:
        print("Summarize URL")
                
        possible_paths = [
            model_path,
            'models/hindi_model.pkl',
            'hindi_model.pkl',
        ]
        from trainer import trainer
        trainer = trainer()
        
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                if summarizer.load_model(path):
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