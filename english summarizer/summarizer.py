import re
import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
import requests
from nltk.stem import WordNetLemmatizer
from collections import defaultdict 
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')  
nltk.download('wordnet')

class EnglishSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentence_tokenizer = PunktSentenceTokenizer()
        self.word_freq = defaultdict(int)
        self.doc_count = defaultdict(int)
        self.total_docs = 0

    def extract_text(self, url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
        
            for element in soup.find_all(['sup', 'span', 'a'], class_=re.compile('reference|citation')):
                element.decompose()
        
            text = ' '.join(p.get_text() for p in soup.find_all(['p', 'article']))
            text = re.sub(r'\[[0-9, ]+\]|\([A-Za-z ]+[0-9]{4}\)', '', text)
            text = re.sub(r'\([^)]*\)', ' ', text)  
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        except Exception as e:
            print(f"Error fetching URL {url}: {str(e)}")
            return None
    
         
    def preprocess(self, text):
        sentences = self.sentence_tokenizer.tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.split()) > 3] 
    
    def compute_tf(self, sentence):
        words = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.lower() not in self.stop_words and word.isalpha() and len(word) > 2]
        tf = defaultdict(float)
        for word in words:
            tf[word] += 1.0 
        if words:
            for word in tf:
                tf[word] /= len(words)
        return tf
    
    def compute_idf(self, sentences):
        self.doc_count = defaultdict(int)
        self.total_docs = len(sentences)

        for sentence in sentences:
            words = set(self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.lower() and word.isalpha() not in self.stop_words)
            for word in words:
                self.doc_count[word] += 1

    def score_sentences(self, sentences):
        self.compute_idf(sentences)
        scores = []

        for i, sent in enumerate(sentences):
            position = i / len(sentences)
            pos_weight = 1 - (2 * abs(0.5 - position))
            
            punc_weight = 1.2 if sent.endswith(('.', '!', '?')) else 1.0

            tf = self.compute_tf(sent)
            score = 0
            for word, freq in tf.items():
                idf = math.log((self.total_docs + 1) / (self.doc_count.get(word,0) + 1)) 
                score += freq * idf 
            
            scores.append(score * pos_weight * punc_weight)

        return scores
    
    def summarize(self, text, num_sentences=5):
        sentences = self.preprocess(text)
        if len(sentences) <= num_sentences:
            return text
        scores = self.score_sentences(sentences)
        ranked = sorted(((scores[i], i) for i in range(len(scores))), reverse=True)

        selected = []
        for score, idx in ranked:
            if len(selected) >= num_sentences:
                break
            if not selected or all(abs(idx - s) > len(sentences)*0.15 for s in selected):
                selected.append(idx)
            selected.sort()
            summary = ' '.join(sentences[i] for i in selected)
            summary = re.sub(r'\s([?.!,;])', r'\1', summary)
            summary = re.sub(r'([?.!])([A-Z])', r'\1 \2', summary)
            summary = summary[0].upper() + summary[1:] 
            if not summary[-1] in '.!?':
                summary = summary.rstrip('.!?,;') + '.'

        return summary
    
    def summarize_url(self, url, num_sentences=5):
        text = self.extract_text(url)
        if not text:
            return "could not extract from URL"
        
        return self.summarize(text, num_sentences)
    
if __name__ == "__main__":
    summarizer = EnglishSummarizer()
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    
    print("Fetching and summarizing URL...")
    summary = summarizer.summarize_url(url, num_sentences=5)
    
    print("\n=== Generated Summary ===")
    print(summary)
    print("\nOriginal URL:", url)