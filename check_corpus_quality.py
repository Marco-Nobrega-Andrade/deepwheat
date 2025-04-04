#!/usr/bin/env python3
"""
Wheat Disease Corpus Quality Checker

This script analyzes a corpus of wheat disease texts to evaluate quality,
relevance, and suitability for NLP tasks like Word2Vec training.
"""

import os
import re
import sys
import argparse
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("corpus_quality.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define disease-related terms for quality evaluation
DISEASE_TERMS = {
    "yellow_rust": ["yellow rust", "stripe rust", "puccinia striiformis", "yr", "pst"],
    "brown_rust": ["brown rust", "leaf rust", "puccinia triticina", "puccinia recondita", "lr"],
    "septoria": ["septoria", "septoria tritici", "zymoseptoria tritici", "septoria leaf blotch", "stb"],
    "mildew": ["mildew", "powdery mildew", "blumeria graminis", "erysiphe graminis", "pm"]
}

# Define management and treatment terms
MANAGEMENT_TERMS = [
    "management", "control", "integrated", "strategy", "practice", "rotation", 
    "treatment", "fungicide", "application", "spray", "timing", "dose", "resistance"
]

# Get stopwords
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

def read_corpus(corpus_file):
    """Read the corpus file and return sentences"""
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences
    except Exception as e:
        logger.error(f"Error reading corpus file: {e}")
        return []

def basic_statistics(sentences):
    """Calculate basic statistics about the corpus"""
    if not sentences:
        return {}
    
    # Calculate word counts
    word_counts = [len(s.split()) for s in sentences]
    
    # Get all words for vocabulary analysis
    all_words = []
    for s in sentences:
        all_words.extend([w.lower() for w in word_tokenize(s) if w.isalpha()])
    
    # Calculate statistics
    stats = {
        "sentence_count": len(sentences),
        "total_word_count": sum(word_counts),
        "vocabulary_size": len(set(all_words)),
        "avg_sentence_length": round(sum(word_counts) / len(sentences), 2),
        "min_sentence_length": min(word_counts),
        "max_sentence_length": max(word_counts),
        "median_sentence_length": round(np.median(word_counts), 2)
    }
    
    return stats

def analyze_domain_relevance(sentences):
    """Analyze how relevant the corpus is to wheat diseases"""
    results = {
        "disease_term_counts": {},
        "management_term_count": 0,
        "disease_coverage": {}
    }
    
    # Count disease term occurrences
    all_text = " ".join(sentences).lower()
    
    # Check disease terms
    for disease, terms in DISEASE_TERMS.items():
        disease_count = 0
        term_counts = {}
        for term in terms:
            count = all_text.count(term)
            term_counts[term] = count
            disease_count += count
        
        results["disease_term_counts"][disease] = term_counts
        results["disease_coverage"][disease] = disease_count
    
    # Check management terms
    for term in MANAGEMENT_TERMS:
        results["management_term_count"] += all_text.count(term)
    
    # Calculate overall relevance score (simple version)
    total_disease_mentions = sum(results["disease_coverage"].values())
    if total_disease_mentions > 0:
        # Calculate balance between diseases
        counts = list(results["disease_coverage"].values())
        max_count = max(counts)
        min_count = min(counts)
        if max_count > 0:
            balance_score = min_count / max_count  # 1.0 = perfectly balanced, 0.0 = completely imbalanced
        else:
            balance_score = 0.0
        
        results["disease_balance"] = round(balance_score, 2)
    else:
        results["disease_balance"] = 0.0
    
    return results

def plot_sentence_length_distribution(sentences, output_dir):
    """Create histogram of sentence lengths"""
    word_counts = [len(s.split()) for s in sentences]
    
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Words per Sentence')
    plt.ylabel('Frequency')
    plt.title('Sentence Length Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics to plot
    avg_len = round(np.mean(word_counts), 2)
    median_len = round(np.median(word_counts), 2)
    plt.axvline(avg_len, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_len}')
    plt.axvline(median_len, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_len}')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, "sentence_length_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def analyze_most_common_terms(sentences, n=50, ngram_range=(1, 1)):
    """Find the most common terms in the corpus"""
    # Create a text corpus with one document per sentence
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        max_features=n
    )
    
    try:
        X = vectorizer.fit_transform(sentences)
        terms = vectorizer.get_feature_names_out()
        
        # Get counts for each term
        term_counts = X.sum(axis=0).A1
        
        # Create dictionary of term -> count
        term_freq = {term: count for term, count in zip(terms, term_counts)}
        
        # Sort by frequency
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_terms
    except Exception as e:
        logger.error(f"Error analyzing common terms: {e}")
        return []

def plot_top_terms(terms, title, output_path):
    """Plot bar chart of top terms"""
    if not terms:
        return None
    
    # Take top 20 or all if less than 20
    top_n = min(20, len(terms))
    terms = terms[:top_n]
    
    # Create dataframe for plotting
    df = pd.DataFrame(terms, columns=['Term', 'Count'])
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df['Term'], df['Count'], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Term')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert to have highest frequency at top
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.0f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def calculate_linguistic_diversity(sentences):
    """Calculate metrics related to linguistic diversity"""
    # Get all words
    all_words = []
    for sentence in sentences:
        words = [word.lower() for word in word_tokenize(sentence) if word.isalpha()]
        all_words.extend(words)
    
    # Calculate type-token ratio (higher means more diverse vocabulary)
    total_words = len(all_words)
    unique_words = len(set(all_words))
    ttr = unique_words / total_words if total_words > 0 else 0
    
    # Calculate frequency distribution of words
    freq_dist = Counter(all_words)
    
    # Calculate percentage of vocabulary that accounts for 50% of text
    sorted_words = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
    cumulative_count = 0
    words_for_half = 0
    
    for word, count in sorted_words:
        cumulative_count += count
        words_for_half += 1
        if cumulative_count >= total_words / 2:
            break
    
    half_percent = words_for_half / unique_words if unique_words > 0 else 0
    
    return {
        "type_token_ratio": round(ttr, 4),
        "percent_vocab_for_half_text": round(half_percent, 4),
        "unique_word_count": unique_words,
        "total_word_count": total_words
    }

def check_corpus_quality(corpus_file, output_dir):
    """Main function to check corpus quality"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read corpus
    logger.info(f"Reading corpus file: {corpus_file}")
    sentences = read_corpus(corpus_file)
    
    if not sentences:
        logger.error("Corpus is empty or could not be read")
        return False
    
    logger.info(f"Found {len(sentences)} sentences in corpus")
    
    # Run analyses
    logger.info("Calculating basic statistics...")
    stats = basic_statistics(sentences)
    
    logger.info("Analyzing domain relevance...")
    relevance = analyze_domain_relevance(sentences)
    
    logger.info("Plotting sentence length distribution...")
    length_plot = plot_sentence_length_distribution(sentences, output_dir)
    
    logger.info("Finding most common terms...")
    unigrams = analyze_most_common_terms(sentences, n=50, ngram_range=(1, 1))
    bigrams = analyze_most_common_terms(sentences, n=30, ngram_range=(2, 2))
    
    logger.info("Plotting top terms...")
    unigram_plot = plot_top_terms(unigrams, "Top Unigrams", os.path.join(output_dir, "top_unigrams.png"))
    bigram_plot = plot_top_terms(bigrams, "Top Bigrams", os.path.join(output_dir, "top_bigrams.png"))
    
    logger.info("Calculating linguistic diversity...")
    diversity = calculate_linguistic_diversity(sentences)
    
    # Show sample sentences
    sample_size = min(5, len(sentences))
    sample_sentences = sentences[:sample_size]
    
    # Combine all results
    results = {
        "basic_stats": stats,
        "domain_relevance": relevance,
        "linguistic_diversity": diversity,
        "sample_sentences": sample_sentences,
        "plots": {
            "sentence_length_distribution": length_plot,
            "top_unigrams": unigram_plot,
            "top_bigrams": bigram_plot
        }
    }
    
    # Save results to file
    results_file = os.path.join(output_dir, "corpus_quality_report.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create a text report for quick review
    text_report = os.path.join(output_dir, "corpus_quality_summary.txt")
    with open(text_report, 'w', encoding='utf-8') as f:
        f.write("=== WHEAT DISEASE CORPUS QUALITY REPORT ===\n\n")
        
        f.write("=== BASIC STATISTICS ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== DOMAIN RELEVANCE ===\n")
        f.write(f"Disease balance score (0-1): {relevance['disease_balance']}\n\n")
        
        f.write("Disease mention counts:\n")
        for disease, count in relevance['disease_coverage'].items():
            f.write(f"- {disease}: {count}\n")
        f.write("\n")
        
        f.write("Management term mentions: " + str(relevance['management_term_count']) + "\n\n")
        
        f.write("=== LINGUISTIC DIVERSITY ===\n")
        for key, value in diversity.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== SAMPLE SENTENCES ===\n")
        for i, sentence in enumerate(sample_sentences, 1):
            f.write(f"{i}. {sentence}\n")
        f.write("\n")
        
        f.write("=== TOP UNIGRAMS ===\n")
        for term, count in unigrams[:10]:
            f.write(f"{term}: {count}\n")
        f.write("\n")
        
        f.write("=== TOP BIGRAMS ===\n")
        for term, count in bigrams[:10]:
            f.write(f"{term}: {count}\n")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Check the quality of a wheat disease corpus")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--output", type=str, default="corpus_quality_report", help="Output directory for reports")
    
    args = parser.parse_args()
    
    check_corpus_quality(args.corpus, args.output)
    
    logger.info("Corpus quality check complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)