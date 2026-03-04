import os
from summary import EnglishSummarizer

def quick_evaluate():
    summarizer = EnglishSummarizer()
    
    model_path = 'english_model.pkl'
    possible_paths = [
            model_path,
            'models/english_model.pkl',
            'english_model.pkl',
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if summarizer.load_model(path):  
                rouge_scores = summarizer.evaluate_on_dataset(
                    dataset_path="/home/banshika/multi-language-news-summarizer/englishsummarizer/datasets",
                    test_split=0.2,
                    num_sentences=3
                )  

                if rouge_scores:
                    print("\nFinal ROUGE Scores:")
                    print(f"ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}")
                    print(f"ROUGE-2 F1: {rouge_scores['rouge2']['fmeasure']:.4f}")
                    print(f"ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}")
                break
    
if __name__ == "__main__":
    quick_evaluate()
    