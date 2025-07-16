#!/usr/bin/env python3
"""
Heat Stress Poultry Research Analysis Script

This script performs comprehensive NLP analysis on heat stress poultry research data including:
1. Data preprocessing and cleaning
2. TF-IDF vectorization with unigrams and bigrams
3. LDA topic modeling
4. Keyword trend analysis over time
5. Topic trend analysis over time
6. Visualization and reporting

Author: GitHub Copilot
Date: July 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuration
DATA_FILE = 'heat_stress_poultry.json'
FIGURES_DIR = 'figures'
N_TOPICS = 7
RANDOM_STATE = 42

# Key terms to track over time
KEY_TERMS = ["selenium", "vitamin e", "probiotics", "antioxidants", "corticosterone", "heat stress"]

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_clean_data(filepath):
    """
    Load JSON data and perform initial cleaning.
    
    Args:
        filepath (str): Path to the JSON data file
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Loading and cleaning data...")
    
    # Load JSON data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display initial data info
    print(f"Initial dataset: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Convert year to integer, handling missing/invalid years
    def convert_year(year_val):
        if pd.isna(year_val) or str(year_val).strip() == '':
            return None
        try:
            return int(str(year_val).strip())
        except (ValueError, TypeError):
            return None
    
    df['year'] = df['year'].apply(convert_year)
    
    # Drop records without abstracts or valid years
    initial_count = len(df)
    df = df.dropna(subset=['abstract', 'year'])
    df = df[df['abstract'].str.strip() != '']
    df = df[(df['year'] >= 1990) & (df['year'] <= 2025)]  # Reasonable year range
    
    print(f"After cleaning: {len(df)} records ({initial_count - len(df)} removed)")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df.reset_index(drop=True)

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, remove stopwords.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    # Add domain-specific stopwords
    stop_words.update(['study', 'studies', 'research', 'result', 'results', 
                      'method', 'methods', 'analysis', 'data', 'showed', 
                      'significantly', 'conclusion', 'background', 'objective'])
    
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def build_bigrams(texts):
    """
    Build bigram phrases from preprocessed texts.
    
    Args:
        texts (list): List of preprocessed texts
        
    Returns:
        list: Texts with bigrams added
    """
    print("Building bigrams...")
    
    # Tokenize all texts
    tokenized_texts = [text.split() for text in texts]
    
    # Find common bigrams
    all_bigrams = []
    for tokens in tokenized_texts:
        all_bigrams.extend([f"{w1}_{w2}" for w1, w2 in bigrams(tokens)])
    
    # Keep bigrams that appear at least 5 times
    bigram_counts = Counter(all_bigrams)
    common_bigrams = {bg for bg, count in bigram_counts.items() if count >= 5}
    
    print(f"Found {len(common_bigrams)} common bigrams")
    
    # Add bigrams to texts
    enriched_texts = []
    for tokens in tokenized_texts:
        text_bigrams = [f"{w1}_{w2}" for w1, w2 in bigrams(tokens) if f"{w1}_{w2}" in common_bigrams]
        enriched_text = ' '.join(tokens + text_bigrams)
        enriched_texts.append(enriched_text)
    
    return enriched_texts

def build_tfidf_matrix(texts):
    """
    Build TF-IDF matrix from preprocessed texts.
    
    Args:
        texts (list): List of preprocessed texts
        
    Returns:
        tuple: (tfidf_matrix, vectorizer, feature_names)
    """
    print("Building TF-IDF matrix...")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 1),  # Unigrams only since we added bigrams manually
        stop_words='english'
    )
    
    # Fit and transform texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(feature_names)}")
    
    return tfidf_matrix, vectorizer, feature_names

def build_lda_model(tfidf_matrix, feature_names, n_topics=7):
    """
    Build LDA topic model using scikit-learn.
    
    Args:
        tfidf_matrix: TF-IDF matrix
        feature_names: Feature names from vectorizer
        n_topics (int): Number of topics
        
    Returns:
        tuple: (lda_model, topic_distributions, topic_labels)
    """
    print(f"Building LDA model with {n_topics} topics...")
    
    # Initialize and fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=20,
        learning_method='batch'
    )
    
    # Fit model and get document-topic distributions
    topic_distributions = lda.fit_transform(tfidf_matrix)
    
    # Print top words for each topic and create labels
    print("\nTop 10 words per topic:")
    print("-" * 50)
    
    topic_labels = []
    n_top_words = 10
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        
        # Create more descriptive labels from top 5 words
        top_5_words = top_words[:5]
        # Create a more readable label by joining words
        label = f"Topic {topic_idx + 1}: {' & '.join(top_5_words[:3]).title()}\n({', '.join(top_5_words[3:5]).lower()})"
        topic_labels.append(label)
    
    return lda, topic_distributions, topic_labels

def analyze_keyword_trends(df, key_terms):
    """
    Analyze keyword trends over time.
    
    Args:
        df (pd.DataFrame): Dataframe with year and abstract columns
        key_terms (list): List of keywords to track
        
    Returns:
        pd.DataFrame: Keyword counts by year
    """
    print("Analyzing keyword trends...")
    
    # Initialize results dictionary
    keyword_data = {'year': []}
    for term in key_terms:
        keyword_data[term] = []
    
    # Get year range
    years = sorted(df['year'].unique())
    
    # Count keyword occurrences per year
    for year in years:
        year_abstracts = df[df['year'] == year]['abstract'].tolist()
        year_text = ' '.join(year_abstracts).lower()
        
        keyword_data['year'].append(year)
        
        for term in key_terms:
            count = year_text.count(term.lower())
            keyword_data[term].append(count)
    
    keyword_df = pd.DataFrame(keyword_data)
    
    return keyword_df

def plot_keyword_trends(keyword_df, key_terms):
    """
    Plot keyword trends over time.
    
    Args:
        keyword_df (pd.DataFrame): Keyword counts by year
        key_terms (list): List of keywords
    """
    print("Plotting keyword trends...")
    
    plt.figure(figsize=(12, 8))
    
    for term in key_terms:
        plt.plot(keyword_df['year'], keyword_df[term], marker='o', 
                linewidth=2, label=term.title(), alpha=0.8)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Mentions', fontsize=12)
    plt.title('Keyword Trends in Heat Stress Poultry Research', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(FIGURES_DIR, 'keyword_trends.pdf'), 
                bbox_inches='tight', dpi=300)
    plt.close()  # Close figure to free memory

def analyze_topic_trends(df, topic_distributions, n_topics):
    """
    Analyze topic trends over time.
    
    Args:
        df (pd.DataFrame): Original dataframe with year column
        topic_distributions (np.array): Document-topic distributions from LDA
        n_topics (int): Number of topics
        
    Returns:
        pd.DataFrame: Topic prevalence by year
    """
    print("Analyzing topic trends...")
    
    # Add topic distributions to dataframe
    topic_df = df.copy()
    for i in range(n_topics):
        topic_df[f'topic_{i+1}'] = topic_distributions[:, i]
    
    # Aggregate by year
    topic_cols = [f'topic_{i+1}' for i in range(n_topics)]
    yearly_topics = topic_df.groupby('year')[topic_cols].mean().reset_index()
    
    return yearly_topics

def plot_topic_trends(yearly_topics, n_topics, topic_labels):
    """
    Plot topic trends over time with dark background and descriptive labels.
    
    Args:
        yearly_topics (pd.DataFrame): Topic prevalence by year
        n_topics (int): Number of topics
        topic_labels (list): Descriptive labels for each topic
    """
    print("Plotting topic trends...")
    
    # Set dark style
    plt.style.use('dark_background')
    
    plt.figure(figsize=(16, 10))
    
    # Use a more vibrant color palette for dark background
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
    
    for i in range(n_topics):
        topic_col = f'topic_{i+1}'
        plt.plot(yearly_topics['year'], yearly_topics[topic_col], 
                marker='o', linewidth=3, label=topic_labels[i], 
                color=colors[i], alpha=0.9, markersize=6)
    
    plt.xlabel('Year', fontsize=14, color='white')
    plt.ylabel('Average Topic Prevalence', fontsize=14, color='white')
    plt.title('Topic Trends in Heat Stress Poultry Research', 
              fontsize=16, fontweight='bold', color='white', pad=20)
    
    # Improved legend with better positioning and formatting
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
              frameon=True, fancybox=True, shadow=True, 
              facecolor='black', edgecolor='white', framealpha=0.8)
    
    # Customize grid
    plt.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Customize axes
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().tick_params(colors='white')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(FIGURES_DIR, 'topic_trends.pdf'), 
                bbox_inches='tight', dpi=300, facecolor='black')
    plt.close()  # Close figure to free memory
    
    # Reset style to default for other plots
    plt.style.use('default')

def print_summary_statistics(df, yearly_topics):
    """
    Print summary statistics and findings.
    
    Args:
        df (pd.DataFrame): Original dataframe
        yearly_topics (pd.DataFrame): Topic trends by year
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS AND FINDINGS")
    print("="*60)
    
    # Top 5 years by number of abstracts
    year_counts = df['year'].value_counts().head()
    print("\nTop 5 years by number of abstracts:")
    print("-" * 40)
    for year, count in year_counts.items():
        print(f"{year}: {count} abstracts")
    
    # Sample topic-year table
    print("\nTopic-Year Distribution Sample (last 5 years):")
    print("-" * 50)
    recent_years = yearly_topics.tail()
    topic_cols = [col for col in yearly_topics.columns if col.startswith('topic_')]
    
    print("Year\t" + "\t".join([f"T{i+1}" for i in range(len(topic_cols))]))
    print("-" * (8 + len(topic_cols) * 8))
    
    for _, row in recent_years.iterrows():
        year = int(row['year'])
        topic_values = [f"{row[col]:.3f}" for col in topic_cols]
        print(f"{year}\t" + "\t".join(topic_values))
    
    print(f"\nTotal abstracts analyzed: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Average abstracts per year: {len(df) / len(df['year'].unique()):.1f}")

def main():
    """
    Main analysis pipeline.
    """
    print("="*60)
    print("HEAT STRESS POULTRY RESEARCH ANALYSIS")
    print("="*60)
    
    # 1. Load and clean data
    df = load_and_clean_data(DATA_FILE)
    
    # 2. Preprocess abstracts
    print("\nPreprocessing abstracts...")
    df['processed_abstract'] = df['abstract'].apply(preprocess_text)
    
    # Remove empty processed abstracts
    df = df[df['processed_abstract'] != ''].reset_index(drop=True)
    print(f"After preprocessing: {len(df)} records")
    
    # 3. Build bigrams and enrich texts
    processed_texts = build_bigrams(df['processed_abstract'].tolist())
    
    # 4. Build TF-IDF matrix
    tfidf_matrix, vectorizer, feature_names = build_tfidf_matrix(processed_texts)
    
    # 5. Build LDA topic model
    lda_model, topic_distributions, topic_labels = build_lda_model(tfidf_matrix, feature_names, N_TOPICS)
    
    # 6. Analyze and plot keyword trends
    keyword_df = analyze_keyword_trends(df, KEY_TERMS)
    plot_keyword_trends(keyword_df, KEY_TERMS)
    
    # 7. Analyze and plot topic trends
    yearly_topics = analyze_topic_trends(df, topic_distributions, N_TOPICS)
    plot_topic_trends(yearly_topics, N_TOPICS, topic_labels)
    
    # 8. Print summary statistics
    print_summary_statistics(df, yearly_topics)
    
    print(f"\nAnalysis complete! Figures saved to '{FIGURES_DIR}/' directory.")
    print("Files generated:")
    print("- keyword_trends.pdf")
    print("- topic_trends.pdf")

if __name__ == "__main__":
    main()
