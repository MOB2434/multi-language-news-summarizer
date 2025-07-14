import re
import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import requests
from collections import defaultdict 
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')  

class EnglishSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.word_freq = defaultdict(int)
        self.doc_count = defaultdict(int)
        self.total_docs = 0
        self.sentence_scores = []

    def extract_text(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for elements in soup(['script', 'style','nav', 'footer','iframe']):
                elements.decompose()
            
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            return self.clean_text(text)
        except  Exception as e:
            print(f"Error fetching URL: {e}")
            return None
    
    def clean_text(self, text):
        text = re.sub(r'\[[0-9]*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()
    
    def compute_tf(self, sentence):
        words = word_tokenize(sentence)
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        tf = defaultdict(int)
        for word in words:
            if word not in self.stop_words:
                tf[word] += 1
        total_words = len(words)
        if words:
            for word in tf:
                tf[word] /= total_words
        return tf
    
    def compute_idf(self, sentences):
        self.doc_count = defaultdict(int)
        self.total_docs = len(sentences)

        for sentence in sentences:  
            words = set(word_tokenize(sentence))
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            for word in words:
                    self.doc_count[word] += 1
    
    def compute_tfidf(self, sentences):
        self.compute_idf(sentences)
        self.sentence_scores = []

        for sentence in sentences:
            tf = self.compute_tf(sentence)
            score = 0.0
            for word, freq in tf.items():
                idf = math.log((self.total_docs + 1) / (self.doc_count.get(word, 0) + 1)) + 1
                score += freq * idf
            self.sentence_scores.append((sentence, score))

    def summarize(self, text, num_sentences=5):
        sentences = sent_tokenize(text)
        if len(sentences) < num_sentences:
            return text
        
        self.compute_tfidf(sentences)

        ranked_sentences = sorted (((score, idx) for idx, score in enumerate(self.sentence_scores)), reverse=True)

        top_indices = sorted([idx for score, idx in ranked_sentences[:num_sentences]])

        return ' '.join(sentences[i] for i in top_indices)
    
    def summarize_url(self, url, num_sentences=5):
        text = self.extract_text(url)
        if text:
            return self.summarize(text, num_sentences)
        else:
            return "Could not generate summary."
        
if __name__ == "__main__":
    summarizer = EnglishSummarizer()
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    summary = summarizer.summarize_url(url, num_sentences=3)

    print("Summary:")
    print("=" * 50)
    print(summary)


