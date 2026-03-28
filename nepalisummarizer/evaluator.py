import os
from summary import NepaliSummarizer

def quick_evaluate():
    summarizer = NepaliSummarizer()
    model_path = 'nepali_model.pkl'
    possible_paths = [
            model_path,
            'models/nepali_model.pkl',
            'nepali_model.pkl',
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if summarizer.load_model(path):  
            
                results = summarizer.evaluate_on_dataset(
                dataset_path="/home/banshika/multi-language-news-summarizer/nepalisummarizer/datasets",
                text_column='article',      
                summary_column='summary',
                num_sentences=8,
                sample_size=100
                )

                for metric, scores in results['rouge_scores'].items():
                    print(f"\n{metric.upper()}:")
                    print(f"  Precision: {scores['precision']:.4f}")
                    print(f"  Recall:    {scores['recall']:.4f}")
                    print(f"  F1-Score:  {scores['fmeasure']:.4f}")

                if results:
                    print("\nFinal ROUGE Scores:")
                    print(f"ROUGE-1 F1: {results['rouge_scores']['rouge1']['fmeasure']:.4f}")
                    print(f"ROUGE-2 F1: {results['rouge_scores']['rouge2']['fmeasure']:.4f}")
                    print(f"ROUGE-L F1: {results['rouge_scores']['rougeL']['fmeasure']:.4f}")
                break
    
if __name__ == "__main__":
    quick_evaluate()
    