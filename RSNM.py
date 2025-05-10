import pandas as pd
from transformers import pipeline
import textwrap

# Load dataset
df = pd.read_csv("news.csv")
df.dropna(subset=["Text"], inplace=True)

# Load pre-trained fake news detector
model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
classifier = pipeline("text-classification", model=model_name, truncation=True)

# Split long article into chunks
def split_into_chunks(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Analyze long article using majority voting
def analyze_article(text):
    chunks = split_into_chunks(text)
    results = classifier(chunks)
    
    fake_count = sum(1 for r in results if r['label'] == 'LABEL_1')
    real_count = len(chunks) - fake_count
    avg_confidence = sum(r['score'] for r in results) / len(results)
    
    final_label = "FAKE" if fake_count > real_count else "REAL"
    
    explanation = ""
    if abs(fake_count - real_count) <= 1:
        explanation = "This article has mixed signals. Please verify from trusted sources."
    elif final_label == "FAKE":
        explanation = "The model detected strong patterns of misinformation or unreliability."
    else:
        explanation = "The article seems consistent with trustworthy news patterns."

    return {
        "chunks": len(chunks),
        "prediction": final_label,
        "confidence": round(avg_confidence * 100, 2),
        "explanation": explanation
    }

# Test function: pick random or user article
def test_on_dataset(index=None):
    if index is None:
        sample = df.sample(1).iloc[0]
    else:
        sample = df.iloc[index]

    print(f"\nTitle: {sample.get('title', 'No Title')}")
    print("-" * 40)
    print(textwrap.shorten(sample['Text'], width=500, placeholder="..."))
    
    result = analyze_article(sample['Text'])
    
    print("\n=== AI Fake News Detection Result ===")
    print(f"Prediction       : {result['prediction']}")
    print(f"Avg Confidence   : {result['confidence']}%")
    print(f"Chunks Processed : {result['chunks']}")
    print("Explanation      :", result['explanation'])

# Example usage:
print("1. Input your own article")
print("2. Test from dataset (random)")
choice = input("Choose (1/2): ").strip()

if choice == "1":
    user_input = input("\nPaste your news article:\n> ")
    result = analyze_article(user_input)
    print("\n=== AI Fake News Detection Result ===")
    print(f"Prediction       : {result['prediction']}")
    print(f"Avg Confidence   : {result['confidence']}%")
    print(f"Chunks Processed : {result['chunks']}")
    print("Explanation      :", result['explanation'])
else:
    test_on_dataset()