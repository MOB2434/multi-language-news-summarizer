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
from collections import defaultdict, Counter
from summary import NepaliSummarizer

class trainer:
    def __init__(self):
        self.trained_data = None
        self.textrank_alpha = 0.95  
        self.mmr_lambda = 0.9 
        self.use_position_bias = True
        self.use_length_filter = True
        self.min_sentence_words = 5  
        self.max_sentence_words = 50 
        self.important_words = set()
        self.word_influence_dict = {}  
        self.trained_patterns = {}     
        self.optimal_weights = {     
            'textrank': 0.35,
            'position': 0.25,
            'length': 0.15,
            'keywords': 0.25
        }
        self.tfidf_vectorizer = None 
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
            
    def load_csv_dataset(self, csv_path, text_column='text', summary_column='summary'):
        
        print(f"Loading dataset from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded: {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
            
            if text_column not in df.columns:
                print(f"Warning: '{text_column}' column not found. Available columns: {df.columns.tolist()}")
                possible_text_cols = ['Article', 'Content','article']
                for col in possible_text_cols:
                    if col in df.columns:
                        text_column = col
                        print(f"Using '{text_column}' as text column")
                        break
            
            if summary_column not in df.columns:
                print(f"Warning: '{summary_column}' column not found.")
                possible_summary_cols = ['Summary','summary']
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
    
    def extract_sentences(self, text):
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'[।!?\.]+', text)
        
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 5:
                cleaned.append(sent)
        
        return cleaned

    def train_from_csv(self, csv_path_or_folder, text_column='text', summary_column='summary'):
        if isinstance(csv_path_or_folder, pd.DataFrame):
            df = csv_path_or_folder
            print(f"Training on provided DataFrame with {len(df)} rows")
        else:
            if os.path.isdir(csv_path_or_folder):
                df = self.load_multiple_csvs(csv_path_or_folder)
            else:
                df = self.load_csv_dataset(csv_path_or_folder, text_column, summary_column)
        
        if df is None or df.empty:
            print("No data available for training")
            return False
        
        print(f"\nTraining model on {len(df)} articles...")
        
        word_frequencies = Counter()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Learning important words"):
            article_text = row['cleaned_text']
            words = article_text.split()
            word_frequencies.update(words)
        
        for word, freq in word_frequencies.most_common(200):
            if word not in self.nepali_stopwords and len(word) > 2:
                self.important_words.add(word)
        
        all_lengths = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Learning sentence patterns"):
            article_text = row['cleaned_text']
            sentences = self.extract_sentences(article_text)
            for sent in sentences:
                all_lengths.append(len(sent.split()))
        
        if all_lengths:
            optimal_length = np.median(all_lengths)
            self.min_sentence_words = max(5, int(optimal_length * 0.5))
            self.max_sentence_words = min(50, int(optimal_length * 2))
        
        self.trained_data = {
            'num_articles': len(df),
            'important_words_count': len(self.important_words),
            'min_sentence_words': self.min_sentence_words,
            'max_sentence_words': self.max_sentence_words
        }
        
        print(f"\n✓ Training complete!")
        print(f"  - Learned {len(self.important_words)} important words")
        print(f"  - Optimal sentence length range: {self.min_sentence_words}-{self.max_sentence_words} words")
        
        self.save_model('nepali_model.pkl')
        
        return True

    def save_model(self, filepath='nepali_model.pkl'):
        model_data = {
            'trained_data': self.trained_data,
            'nepali_stopwords': self.nepali_stopwords,
            'important_words': self.important_words
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
            self.important_words = model_data.get('important_words', set())
            
            # Load training parameters if they exist
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
    trainer_instance = trainer()

    summarizer = NepaliSummarizer()

    dataset_path = "/home/banshika/multi-language-news-summarizer/nepalisummarizer/datasets" 

    trainer_instance.train_from_csv(csv_path_or_folder=dataset_path,
                text_column='article',
                summary_column='summary') 
    
if __name__ == "__main__":
    main()