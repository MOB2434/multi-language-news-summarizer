import re
import math
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
import requests
from nltk.stem import WordNetLemmatizer
from collections import defaultdict 
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

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
        self.news_stopwords = {'said', 'says', 'according', 'reported', 'published'} | self.stop_words

    def extract_text(self, url):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, 'html.parser')
        
            #remove citations
            for element in soup.find_all(['sup', 'span', 'a'], class_=re.compile('reference|citation')):
                element.decompose()

            #remove non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'figure', 'blockquote', 'ul', 'ol', 'table', 'form']):
                element.decompose()

            #remove journalistic bylines and timestamps
            for byline in soup.find_all(class_=re.compile('byline|author|timestamp|dateline|published')):
                byline.decompose()  
            for time_tag in soup.find_all(['time',]):
                time_tag.decompose()

            #extract main content
            article_body= soup.find('article') or soup.find(class_=re.compile('article|post|story|content|main|body|section'))
            if not article_body:
                article_body = soup

            paragraphs = []
            for p in article_body.find_all(['p', 'h2', 'h3'], recursive=True):
                text = p.get_text(' ', strip=True)
                if len(text.split()) > 10 and not any(x in text.lower() for x in ['sign up', 'subscribe', 'follow us']):
                    paragraphs.append(text)
            full_text = ' '.join(paragraphs)

            text = re.sub(r'\[[0-9, ]+\]|\([A-Za-z ]+[0-9]{4}\)', '', full_text) 
            text = re.sub(r'\([^)]*\)', ' ', full_text)  
            text = re.sub(r'\s+', ' ', full_text).strip()

            return full_text if full_text else None
        
        except Exception as e:
            print(f"Error fetching URL {url}: {str(e)}")
            return None
             
    def preprocess(self, text):
        text = re.sub(r'^[A-Z\s]+\s?[-–—]\s?', '', text) 
        sentences = []
        for sent in sent_tokenize(text):
            if any(phrase in sent.lower() for phrase in ['photo:', 'image:', 'video:', 'read more:', 'continue reading']):
                continue
            sent = re.sub(r'["“”]', '', sent)
            
            if len(sent.split()) > 5: 
                sentences.append(sent)
        return sentences

    def compute(self, sentences):
        self.doc_count = defaultdict(int)
        self.total_docs = len(sentences)

        #compute idf
        for sentence in sentences:
            words = set(self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.lower() and word.isalpha() not in self.news_stopwords)
            for word in words:
                self.doc_count[word] += 1

        #headline detection
        scores = []
        for i, sent in enumerate(sentences):
            is_headline = i < max(3, len(sentences)*0.1)
            words = [self.lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sent) if w.isalpha() and w.lower() not in self.news_stopwords]
            words = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.lower() not in self.stop_words and word.isalpha() and len(word) > 2]
        
            #compute tf
            tf = defaultdict(float)
            for word in words:
                if word[0].isupper():  
                    tf[word] += 1.5
                else:
                    tf[word] += 1
                    
                if words:
                    max_freq = max(tf.values())
                    for word in tf:
                        tf[word] = 0.5 + 0.5 * (tf[word] / max_freq)
    
            #score sentences
            position = i / len(sentences)
            pos_weight = 1.6 if is_headline else (1.5 - abs(1 - 2 * position))
            score = sum(tf[word] * math.log((self.total_docs + 1) / (self.doc_count.get(word, 0) + 1)) for word in tf)*pos_weight
            scores.append(score)
        return scores

    def summarize(self, text, num_sentences=5):
        sentences = self.preprocess(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        scores = self.compute(sentences)
        ranked = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)

        selected = []
        for score, idx in ranked:
            if len(selected) >= num_sentences:
                break
            if not selected and idx > len(sentences)*0.3:
                continue
            if not selected or all(abs(idx - s) > len(sentences)*0.15 for s in selected):
                selected.append(idx)
            
        selected.sort()
        summary = ' '.join(sentences[i] for i in selected)

        summary = re.sub(r'\s([,.;!?])', r'\1', summary)
        summary = summary[0].upper() + summary[1:]
        if not summary.endswith(('.', '!', '?')):
            summary = summary.rstrip(' ,;') + '.'

        return summary
    
    def summarize_url(self):
        print("News Summarizer")
        print("Type 'quit' to exit\n")
        
        while True:
            url = input("\nEnter URL: ").strip()
            if url.lower() in ('quit', 'exit'):
                break
                
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            print("\n Processing...", end=' ', flush=True)
            text = self.extract_text(url)
            
            if text:
                summary = self.summarize(text)
                print("\n Summary:")
                print("-" * 50)
                print(summary)
                print("-" * 50)
                print(f"\nSource: {url}")
            else:
                print("\n Could not extract content from this URL")
            
            print("\n" + "=" * 50)

if __name__ == "__main__":
    summarizer = EnglishSummarizer()
    summarizer.summarize_url()

       
    
    
