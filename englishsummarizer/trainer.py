import pandas as pd
import numpy as np
import os
import glob
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize


class trainer:
    def __init__(self):
        self.ml_model = None
        self.vectorizer = None
        self.scaler = None
        self.trained_data = None

    def load_csv_dataset(self, csv_path, text_column='text', summary_column='summary'):
            
            print(f"Loading dataset from: {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
                print(f"Dataset loaded: {len(df)} rows")
                print(f"Columns: {df.columns.tolist()}")
                
                if text_column not in df.columns:
                    print(f"Warning: '{text_column}' column not found. Available columns: {df.columns.tolist()}")
                    possible_text_cols = ['article','description','text','content','body']
                    for col in possible_text_cols:
                        if col in df.columns:
                            text_column = col
                            print(f"Using '{text_column}' as text column")
                            break
                
                if summary_column not in df.columns:
                    print(f"Warning: '{summary_column}' column not found.")
                    possible_summary_cols = [ 'ctext','highlights','summary','abstract','description']
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
        """
        Train ML model for extractive summarization
        """
        # Load data
        if os.path.isdir(csv_path_or_folder):
            df = self.load_multiple_csvs(csv_path_or_folder)
        else:
            df = self.load_csv_dataset(csv_path_or_folder, text_column, summary_column)
        
        if df is None or df.empty:
            print("No data available for training")
            return False
        
        print("\nPreparing training data...")
        
        # Extract sentence-level features from training data
        X_features = []
        y_scores = []
        
        for idx, row in tqdm(df.iterrows(), total=min(1000, len(df)), desc="Processing articles"):
            article = row['cleaned_text']
            reference_summary = row.get('cleaned_summary', '')
            
            if len(article.split()) < 50:
                continue
                
            sentences = sent_tokenize(article)
            if len(sentences) < 3:
                continue
                
            # Create sentence embeddings and features
            sent_features = self._extract_sentence_features(sentences)
            
            # Calculate importance scores for each sentence using reference summary
            if reference_summary and len(reference_summary.split()) > 10:
                sent_scores = self._calculate_sentence_importance(sentences, reference_summary)
            else:
                # Use heuristic-based scoring if no reference summary
                sent_scores = self._calculate_heuristic_scores(sentences)
            
            X_features.extend(sent_features)
            y_scores.extend(sent_scores)
            
            if len(X_features) > 50000:  # Limit training data size
                break
        
        if len(X_features) == 0:
            print("No valid training samples found")
            return False
        
        print(f"\nTraining on {len(X_features)} sentence samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train ML model
        print("Training Gradient Boosting Regressor...")
        self.ml_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.ml_model.fit(X_scaled, y_scores)
        
        # Train TF-IDF vectorizer for fallback
        print("\nTraining TF-IDF vectorizer (fallback)...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        all_texts = df['cleaned_text'].tolist()[:1000]  # Limit for memory
        self.vectorizer.fit(all_texts)
        
        # Store training metadata
        self.trained_data = {
            'num_articles': len(df),
            'num_sentences_trained': len(X_features),
            'avg_article_length': np.mean([len(art.split()) for art in df['cleaned_text'].tolist()[:100]]),
            'vocab_size': len(self.vectorizer.vocabulary_),
            'model_type': 'GradientBoostingRegressor',
            'feature_dim': X_scaled.shape[1],
            'columns_used': {
                'text': text_column,
                'summary': summary_column if summary_column in df.columns else None
            }
        }
        
        print(f"\nTraining completed successfully!")
        print(f"  • Articles processed: {self.trained_data['num_articles']}")
        print(f"  • Sentences trained on: {self.trained_data['num_sentences_trained']}")
        print(f"  • Feature dimension: {self.trained_data['feature_dim']}")
        print(f"  • Model: {self.trained_data['model_type']}")
        
        self.save_model('english_model.pkl')
        return True

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

    def _calculate_sentence_importance(self, sentences, reference_summary):
        """
        Calculate importance scores for sentences based on reference summary
        """
        scores = []
        reference_sents = sent_tokenize(reference_summary)
        
        for sent in sentences:
            sent_lower = sent.lower()
            max_similarity = 0
            
            for ref_sent in reference_sents:
                # Simple word overlap similarity
                sent_words = set(word_tokenize(sent_lower))
                ref_words = set(word_tokenize(ref_sent.lower()))
                
                if sent_words and ref_words:
                    overlap = len(sent_words & ref_words)
                    similarity = overlap / max(len(sent_words), len(ref_words))
                    max_similarity = max(max_similarity, similarity)
            
            scores.append(max_similarity)
        
        # Normalize scores
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]
        
        return scores

    def _calculate_heuristic_scores(self, sentences):
        """
        Calculate heuristic-based scores when no reference summary available
        """
        scores = []
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            for i in range(len(sentences)):
                # Centrality score
                centrality = similarity_matrix[i].sum() / len(sentences)
                
                # Position bonus
                position = 1.0 if i < 3 else (0.8 if i > len(sentences) - 2 else 0.5)
                
                # Length score
                word_count = len(sentences[i].split())
                length_score = min(word_count / 30, 1.0)
                
                # Combined score
                score = centrality * 0.5 + position * 0.3 + length_score * 0.2
                scores.append(score)
        except:
            # Fallback to simple position-based scoring
            for i in range(len(sentences)):
                scores.append(1.0 if i < 3 else (0.8 if i > len(sentences) - 2 else 0.5))
        
        return scores

    def save_model(self, filepath='english_model.pkl'):
        """
        Save trained model including ML model
        """
        if self.ml_model is not None or self.vectorizer is not None:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'ml_model': self.ml_model,
                    'scaler': self.scaler,
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
                self.vectorizer = data.get('vectorizer')
                self.ml_model = data.get('ml_model')
                self.scaler = data.get('scaler')
                self.trained_data = data.get('trained_data')
            print(f"Model loaded from {filepath}")
            if self.trained_data:
                print(f"  • Model type: {self.trained_data.get('model_type', 'Unknown')}")
                print(f"  • Trained on {self.trained_data.get('num_articles', 0)} articles")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
def main():
    trainer_instance = trainer()
    from summary import EnglishSummarizer

    summarizer = EnglishSummarizer()

    dataset_path = "/home/banshika/multi-language-news-summarizer/englishsummarizer/datasets" 

    trainer_instance.train_from_csv(csv_path_or_folder=dataset_path,
                text_column='article',
                summary_column='summary') 
    
if __name__ == "__main__":
    main()