import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup

class FakeNewsDetector:
    def __init__(self):
        self.model_name = "Pulk17/Fake-News-Detection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def extract(self, url):
        response = requests.get(url, timeout=10)
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

        article = soup.find('article') or soup.find(class_=re.compile('article|post|story|content|main|body|section'))
            
        if article:
            text = article.get_text()
        else:
            text = soup.find('body').get_text()
        
        return text
    
    def detect(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

        id2label = self.model.config.id2label 
        if not id2label:
            id2label = {0: "FAKE", 1: "REAL"}

        predicted_label = id2label.get(predicted_class_id, f"Unknown (ID: {predicted_class_id})")
        probability = torch.nn.functional.softmax(logits, dim=-1)[0]

        if probability[0].item() > probability[1].item():
            return "The article is likely fake."
        else:    
            return "The article is likely real."
        
def main():
    url = input("Enter the URL of the news article: ")
    detector = FakeNewsDetector()
    
    text = detector.extract(url)
    result = detector.detect(text)
    print(result)
    
if __name__ == "__main__":
    main()
