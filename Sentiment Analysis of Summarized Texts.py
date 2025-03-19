# Step 1: Installing the required libraries
import subprocess
import sys

def install_packages():
    packages = ['transformers', 'nltk', 'scikit-learn', 'datasets', 'rouge-score']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installing the required libraries
install_packages()

########################################
# Step 2: Import libraries and download datasets
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer # type: ignore
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a list of languages
languages = ['english', 'german', 'french', 'spanish', 'italian', 'finnish', 'hungarian', 'arabic']

# Load datasets from open sources
datasets = {
    'english': load_dataset('imdb'),  # English dataset
    'german': load_dataset('germeval', '2018'),  # German dataset
    'french': load_dataset('semeval', '2018.task1'),  # French dataset
    'spanish': load_dataset('semeval', '2018.task1'),  # Spanish dataset
    'italian': load_dataset('semeval', '2018.task1'),  # Italian dataset
    'finnish': load_dataset('turkunlp/finnish_sentiment'),  # Finnish dataset
    'hungarian': load_dataset('turkunlp/hungarian_emotion'),  # Hungarian dataset
    'arabic': load_dataset('arbml/arsas')  # Arabic dataset
}

# Check the structure of the datasets
for lang, dataset in datasets.items():
    print(f"Language: {lang}")
    print(f"Dataset: {dataset}")
    print(f"Sample data: {dataset['train'][0]}")
    print("\n")

########################################
# Step 3: Data preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text, language):
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text, language=language)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Example of text preprocessing
for lang, dataset in datasets.items():
    dataset['train'] = dataset['train'].map(lambda x: {'text': preprocess_text(x['text'], lang)})

########################################
# Step 4: Extractive summation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extractive_summarization(text, ratio=0.3):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores, axis=0)[::-1]]
    summary = ' '.join(ranked_sentences[:int(len(ranked_sentences) * ratio)])
    return summary

# Example of extractive summarization
for lang, dataset in datasets.items():
    dataset['train'] = dataset['train'].map(lambda x: {'summary': extractive_summarization(x['text'])})

########################################
# Step 5: Abstract Summarization
def abstractive_summarization(text, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example of abstract summarization
for lang, dataset in datasets.items():
    dataset['train'] = dataset['train'].map(lambda x: {'summary': abstractive_summarization(x['text'])})

########################################
# Step 6: Sentiment Analysis
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Example of sentiment analysis
for lang, dataset in datasets.items():
    dataset['train'] = dataset['train'].map(lambda x: {'sentiment': analyze_sentiment(x['text'])[0]})

########################################
# Step 7: Evaluate the results
def evaluate_sentiment(original_sentiments, summarized_sentiments):
    accuracy = accuracy_score(original_sentiments, summarized_sentiments)
    precision = precision_score(original_sentiments, summarized_sentiments, average='weighted')
    recall = recall_score(original_sentiments, summarized_sentiments, average='weighted')
    f1 = f1_score(original_sentiments, summarized_sentiments, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_summary(original_texts, summarized_texts):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(original, summary) for original, summary in zip(original_texts, summarized_texts)]
    rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    return rouge1, rouge2, rougeL

# Example of evaluation
for lang, dataset in datasets.items():
    original_sentiments = [x['sentiment'] for x in dataset['train']]
    summarized_sentiments = [analyze_sentiment(x['summary'])[0] for x in dataset['train']]
    accuracy, precision, recall, f1 = evaluate_sentiment(original_sentiments, summarized_sentiments)
    rouge1, rouge2, rougeL = evaluate_summary([x['text'] for x in dataset['train']], [x['summary'] for x in dataset['train']])
    print(f"Language: {lang}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(f"ROUGE-1: {rouge1}, ROUGE-2: {rouge2}, ROUGE-L: {rougeL}")

########################################
# Step 8: Visualizing Results
import matplotlib.pyplot as plt  # type: ignore

# Define a list of languages
languages ​​= list(datasets.keys())

# Calculate accuracy for each language
accuracies = [
    evaluate_sentiment(
        [x['sentiment'] for x in datasets[lang]['train']],
        [analyze_sentiment(x['summary'])[0] for x in datasets[lang]['train']]
    )[0]
    for lang in languages
]

# Building a graph
plt.figure(figsize=(10, 6))
plt.bar(languages, accuracies, color='blue')
plt.xlabel('Language')
plt.ylabel('Accuracy')
plt.title('Sentiment Analysis Accuracy After Summarization')
plt.show()