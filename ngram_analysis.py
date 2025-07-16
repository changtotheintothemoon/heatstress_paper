#!/usr/bin/env python3
"""
N-gram Analysis Script for Heat Stress Poultry Research

This script performs comprehensive N-gram analysis on PubMed abstracts including:
1. Text preprocessing and cleaning
2. POS tagging to filter meaningful words
3. Unigram, bigram, and trigram frequency analysis
4. Tabular output of top N-grams
5. NetworkX visualization of top bigrams

Author: GitHub Copilot
Date: July 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import string
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.util import bigrams, trigrams

# Download required NLTK data
required_nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
for data in required_nltk_data:
    try:
        if data == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif data == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif data == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print(f"Downloading {data}...")
        nltk.download(data)

# Load spaCy model (try to use en_core_web_sm, fallback to basic processing)
try:
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
    print("Using spaCy for POS tagging")
except OSError:
    print("spaCy model not found, using NLTK for POS tagging")
    USE_SPACY = False

def load_abstracts(json_file):
    """Load abstracts from JSON file."""
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    abstracts = []
    for record in data:
        abstract = record.get('abstract', '').strip()
        if abstract:  # Only include non-empty abstracts
            abstracts.append(abstract)
    
    print(f"Loaded {len(abstracts)} abstracts with content")
    return abstracts

def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation, numbers, and special characters
    - Removing extra whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def pos_filter_tokens(tokens):
    """
    Filter tokens using POS tagging to keep only nouns, verbs, and adjectives.
    """
    if USE_SPACY:
        # Join tokens back to text for spaCy processing
        text = ' '.join(tokens)
        doc = nlp(text)
        
        # Keep nouns, verbs, adjectives
        filtered_tokens = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2):
                filtered_tokens.append(token.lemma_)
        
        return filtered_tokens
    else:
        # Use NLTK POS tagging
        pos_tags = pos_tag(tokens)
        
        # Keep nouns (NN*), verbs (VB*), adjectives (JJ*)
        target_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
        
        filtered_tokens = []
        for word, pos in pos_tags:
            if pos in target_pos and len(word) > 2:
                filtered_tokens.append(word)
        
        return filtered_tokens

def process_abstracts(abstracts):
    """
    Process all abstracts through the complete preprocessing pipeline.
    """
    print("Preprocessing abstracts...")
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add domain-specific stopwords
    domain_stopwords = {
        'study', 'studies', 'research', 'analysis', 'result', 'results', 
        'conclusion', 'abstract', 'background', 'method', 'methods',
        'objective', 'purpose', 'aim', 'significantly', 'showed', 'observed',
        'found', 'compared', 'treatment', 'treatments', 'control', 'group',
        'groups', 'effect', 'effects', 'level', 'levels', 'high', 'low',
        'day', 'days', 'week', 'weeks', 'respectively', 'however', 'therefore',
        'thus', 'although', 'moreover', 'furthermore', 'additionally'
    }
    stop_words.update(domain_stopwords)
    
    all_tokens = []
    
    for i, abstract in enumerate(abstracts):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(abstracts)} abstracts")
        
        # Preprocess text
        cleaned_text = preprocess_text(abstract)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # POS filtering
        filtered_tokens = pos_filter_tokens(tokens)
        
        all_tokens.extend(filtered_tokens)
    
    print(f"Total tokens after preprocessing: {len(all_tokens)}")
    return all_tokens

def generate_ngrams(tokens):
    """Generate unigrams, bigrams, and trigrams from tokens."""
    print("Generating N-grams...")
    
    # Unigrams
    unigrams = tokens
    
    # Bigrams
    bigram_list = list(bigrams(tokens))
    
    # Trigrams
    trigram_list = list(trigrams(tokens))
    
    print(f"Generated {len(unigrams)} unigrams, {len(bigram_list)} bigrams, {len(trigram_list)} trigrams")
    
    return unigrams, bigram_list, trigram_list

def count_ngrams(unigrams, bigram_list, trigram_list):
    """Count frequency of each N-gram type."""
    print("Counting N-gram frequencies...")
    
    # Count frequencies
    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigram_list)
    trigram_counts = Counter(trigram_list)
    
    print(f"Unique unigrams: {len(unigram_counts)}")
    print(f"Unique bigrams: {len(bigram_counts)}")
    print(f"Unique trigrams: {len(trigram_counts)}")
    
    return unigram_counts, bigram_counts, trigram_counts

def print_top_ngrams(unigram_counts, bigram_counts, trigram_counts, top_n=50):
    """Print top N-grams in tabular format."""
    
    print("\n" + "="*80)
    print("TOP N-GRAM ANALYSIS RESULTS")
    print("="*80)
    
    # Top Unigrams
    print(f"\nTOP {top_n} UNIGRAMS")
    print("-" * 50)
    print(f"{'Rank':<6} {'Word':<20} {'Frequency':<12} {'%':<8}")
    print("-" * 50)
    
    total_unigrams = sum(unigram_counts.values())
    for i, (word, count) in enumerate(unigram_counts.most_common(top_n), 1):
        percentage = (count / total_unigrams) * 100
        print(f"{i:<6} {word:<20} {count:<12} {percentage:.2f}%")
    
    # Top Bigrams
    print(f"\nTOP {top_n} BIGRAMS")
    print("-" * 60)
    print(f"{'Rank':<6} {'Bigram':<30} {'Frequency':<12} {'%':<8}")
    print("-" * 60)
    
    total_bigrams = sum(bigram_counts.values())
    for i, (bigram, count) in enumerate(bigram_counts.most_common(top_n), 1):
        bigram_str = ' '.join(bigram)
        percentage = (count / total_bigrams) * 100
        print(f"{i:<6} {bigram_str:<30} {count:<12} {percentage:.2f}%")
    
    # Top Trigrams
    print(f"\nTOP {top_n} TRIGRAMS")
    print("-" * 70)
    print(f"{'Rank':<6} {'Trigram':<40} {'Frequency':<12} {'%':<8}")
    print("-" * 70)
    
    total_trigrams = sum(trigram_counts.values())
    for i, (trigram, count) in enumerate(trigram_counts.most_common(top_n), 1):
        trigram_str = ' '.join(trigram)
        percentage = (count / total_trigrams) * 100
        print(f"{i:<6} {trigram_str:<40} {count:<12} {percentage:.2f}%")

def create_bigram_network(bigram_counts, top_n=30):
    """
    Create and visualize a NetworkX graph of top bigrams.
    Nodes are words, edges are bigram relationships with weights as frequencies.
    """
    print(f"\nCreating network visualization of top {top_n} bigrams...")
    
    # Get top bigrams
    top_bigrams = bigram_counts.most_common(top_n)
    
    # Create graph
    G = nx.Graph()
    
    # Add edges (bigrams) with weights (frequencies)
    for (word1, word2), count in top_bigrams:
        G.add_edge(word1, word2, weight=count)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes
    node_sizes = [G.degree(node) * 300 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    edge_widths = [weight / max_weight * 5 for weight in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f'Bigram Network Analysis - Top {top_n} Bigrams\n'
              f'Node size = degree centrality, Edge thickness = frequency',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('figures/bigram_network.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/bigram_network.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved as 'figures/bigram_network.pdf' and 'figures/bigram_network.png'")
    
    # Print network statistics
    print(f"\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    
    # Top nodes by degree centrality
    centrality = nx.degree_centrality(G)
    top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\nTop 10 nodes by degree centrality:")
    for i, (node, cent) in enumerate(top_central_nodes, 1):
        print(f"{i:2d}. {node:<15} {cent:.3f}")
    
    plt.show()
    
    return G

def main():
    """Main function to run the complete N-gram analysis."""
    
    print("HEAT STRESS POULTRY N-GRAM ANALYSIS")
    print("=" * 50)
    
    # Load data
    abstracts = load_abstracts('heat_stress_poultry.json')
    
    # Process abstracts
    tokens = process_abstracts(abstracts)
    
    # Generate N-grams
    unigrams, bigram_list, trigram_list = generate_ngrams(tokens)
    
    # Count frequencies
    unigram_counts, bigram_counts, trigram_counts = count_ngrams(unigrams, bigram_list, trigram_list)
    
    # Print results
    print_top_ngrams(unigram_counts, bigram_counts, trigram_counts, top_n=50)
    
    # Create network visualization
    create_bigram_network(bigram_counts, top_n=30)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Total abstracts processed: {len(abstracts)}")
    print(f"Total tokens after preprocessing: {len(tokens)}")
    print(f"Unique unigrams: {len(unigram_counts)}")
    print(f"Unique bigrams: {len(bigram_counts)}")
    print(f"Unique trigrams: {len(trigram_counts)}")
    print("\nFiles generated:")
    print("- figures/bigram_network.pdf")
    print("- figures/bigram_network.png")

if __name__ == "__main__":
    main()
