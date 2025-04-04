#!/usr/bin/env python3
"""
Wheat Disease Word2Vec Model
============================

Specialized Word2Vec model focusing on 4 key wheat diseases:
- Yellow rust
- Brown rust
- Septoria
- Powdery mildew

With emphasis on yield loss metrics and treatment approaches.
"""

import os
import re
import logging
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wheat_disease_model.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define stopwords with domain-specific adjustments
STOPWORDS = set(stopwords.words('english'))
# Remove important domain terms from stopwords
DOMAIN_TERMS = {
    'yellow', 'rust', 'brown', 'leaf', 'powdery', 'mildew', 'septoria',
    'fungicide', 'wheat', 'disease', 'infection', 'yield', 'loss',
    'treatment', 'application', 'control', 'severity', 'resistance'
}
STOPWORDS = STOPWORDS - DOMAIN_TERMS

# Define our target diseases and related terms
TARGET_DISEASES = {
    'yellow_rust': ['yellow rust', 'stripe rust', 'puccinia striiformis'],
    'brown_rust': ['brown rust', 'leaf rust', 'puccinia triticina', 'puccinia recondita'],
    'septoria': ['septoria', 'septoria tritici', 'zymoseptoria tritici', 'mycosphaerella graminicola', 'septoria blotch'],
    'mildew': ['powdery mildew', 'blumeria graminis', 'erysiphe graminis']
}

# Management and yield terminology
MANAGEMENT_TERMS = [
    'fungicide application', 'fungicide treatment', 'seed treatment',
    'resistant variety', 'resistant varieties', 'disease management',
    'disease control', 'chemical control', 'integrated management',
    'spray', 'spraying', 'timing', 'application timing', 'dose',
    'rate', 'treatment interval'
]

YIELD_TERMS = [
    'yield loss', 'yield reduction', 'grain yield', 'quality loss',
    'thousand kernel weight', 'test weight', 'grain quality',
    'economic loss', 'harvest loss', 'production loss'
]

# Combine all multi-word terms we want to preserve
MULTI_WORD_TERMS = []
for terms in TARGET_DISEASES.values():
    MULTI_WORD_TERMS.extend(terms)
MULTI_WORD_TERMS.extend(MANAGEMENT_TERMS)
MULTI_WORD_TERMS.extend(YIELD_TERMS)

class WheatDiseaseModel:
    def __init__(self, corpus_file, model_dir="models"):
        """
        Initialize the wheat disease model
        
        Args:
            corpus_file: Path to the corpus text file
            model_dir: Directory to save trained models
        """
        self.corpus_file = corpus_file
        self.model_dir = model_dir
        self.sentences = []
        self.processed_sentences = []
        self.model = None
        
        # Compile patterns for multi-word term detection
        self.term_patterns = self._compile_term_patterns()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def _compile_term_patterns(self):
        """Compile regex patterns for multi-word term detection"""
        # Sort terms by length (to match longer terms first)
        sorted_terms = sorted(MULTI_WORD_TERMS, key=len, reverse=True)
        
        # Escape special regex characters and convert to patterns
        patterns = []
        for term in sorted_terms:
            # Convert spaces to flexible whitespace matcher
            pattern = re.escape(term).replace('\ ', '\\s+')
            # Compile case-insensitive pattern
            patterns.append(re.compile(r'\b' + pattern + r'\b', re.IGNORECASE))
            
        return patterns
        
    def load_corpus(self):
        """Load and preprocess the corpus"""
        logger.info(f"Loading corpus from {self.corpus_file}")
        
        # Load sentences from corpus file
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(self.sentences)} sentences")
        return self.sentences
    
    def _preserve_multi_word_terms(self, text):
        """
        Identify and temporarily mark multi-word terms to preserve them during tokenization
        Returns text with multi-word terms marked with special tokens
        """
        marked_text = text.lower()
        
        # Dictionary to track replacements for later restoration
        replacements = {}
        
        # Replace multi-word terms with temporary placeholders
        for i, pattern in enumerate(self.term_patterns):
            matches = list(pattern.finditer(marked_text))
            
            # Process matches from end to start to avoid messing up indices
            for match in reversed(matches):
                start, end = match.span()
                original = marked_text[start:end]
                # Create a unique placeholder that won't be in the text
                placeholder = f"__MULTIWORD_TERM_{i}_{len(replacements)}__"
                # Track the replacement
                replacements[placeholder] = original
                # Replace in the text
                marked_text = marked_text[:start] + placeholder + marked_text[end:]
        
        return marked_text, replacements
    
    def _restore_multi_word_terms(self, tokens, replacements):
        """Restore multi-word terms in tokenized text"""
        restored_tokens = []
        
        for token in tokens:
            if token in replacements:
                # Split the original term into component words
                term_words = replacements[token].split()
                restored_tokens.extend(term_words)
            else:
                restored_tokens.append(token)
                
        return restored_tokens
    
    def preprocess_corpus(self):
        """
        Preprocess the corpus for Word2Vec training, preserving multi-word terms
        
        Returns:
            List of tokenized sentences
        """
        logger.info("Preprocessing corpus")
        if not self.sentences:
            self.load_corpus()
            
        processed_sentences = []
        
        # Track which multi-word terms were found
        found_terms = {}
        
        for sentence in tqdm(self.sentences, desc="Processing sentences"):
            # Preserve multi-word terms
            marked_text, replacements = self._preserve_multi_word_terms(sentence)
            
            # Update found terms
            for placeholder, original in replacements.items():
                orig_lower = original.lower()
                if orig_lower not in found_terms:
                    found_terms[orig_lower] = 0
                found_terms[orig_lower] += 1
            
            # Tokenize the marked text
            tokens = word_tokenize(marked_text)
            
            # Restore multi-word terms as separate tokens
            tokens = self._restore_multi_word_terms(tokens, replacements)
            
            # Remove stopwords and punctuation
            filtered_tokens = []
            for token in tokens:
                if token.isalpha() and token not in STOPWORDS and len(token) > 1:
                    filtered_tokens.append(token)
                    
            if filtered_tokens:
                processed_sentences.append(filtered_tokens)
        
        # Log which multi-word terms were found
        if found_terms:
            logger.info(f"Found {len(found_terms)} multi-word terms in corpus")
            # List the top found terms
            top_terms = sorted(found_terms.items(), key=lambda x: x[1], reverse=True)[:20]
            logger.info("Top multi-word terms:")
            for term, count in top_terms:
                logger.info(f"  {term}: {count} occurrences")
                
        # Check if our target disease terms were found
        for disease, terms in TARGET_DISEASES.items():
            found = sum(term.lower() in found_terms for term in terms)
            if found:
                logger.info(f"Found {found}/{len(terms)} variants for {disease}")
            else:
                logger.warning(f"No terms found for {disease}!")
            
        self.processed_sentences = processed_sentences
        logger.info(f"Preprocessed {len(processed_sentences)} sentences for training")
        
        return processed_sentences
    
    def train_model(self, vector_size=200, window=5, min_count=5, epochs=15, workers=None):
        """
        Train a Word2Vec model on the preprocessed corpus
        
        Args:
            vector_size: Dimensionality of the word vectors
            window: Max distance between the current and predicted word
            min_count: Ignore words with frequency below this
            epochs: Number of training epochs
            workers: Number of worker threads to use
            
        Returns:
            Trained Word2Vec model
        """
        if not self.processed_sentences:
            self.preprocess_corpus()
            
        # Set workers to the number of available cores if not specified
        if workers is None:
            workers = multiprocessing.cpu_count()
            
        logger.info(f"Training Word2Vec model with parameters: "
                   f"vector_size={vector_size}, window={window}, min_count={min_count}, "
                   f"epochs={epochs}, workers={workers}")
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=self.processed_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1  # Skip-gram model is typically better for specialized domains
        )
        
        vocab_size = len(self.model.wv.key_to_index)
        logger.info(f"Model trained with vocabulary size of {vocab_size}")
        return self.model
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return None
            
        if filename is None:
            filename = "wheat_disease_model.model"
            
        model_path = os.path.join(self.model_dir, filename)
        self.model.save(model_path)
        
        # Also save word vectors in word2vec format for compatibility
        wv_path = os.path.join(self.model_dir, f"{os.path.splitext(filename)[0]}.wordvectors")
        self.model.wv.save(wv_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Word vectors saved to {wv_path}")
        
        return model_path
    
    def load_model(self, filename):
        """Load a previously trained model"""
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return None
            
        self.model = Word2Vec.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        return self.model
        
    def find_similar_terms(self, term, n=10):
        """Find terms similar to a given term"""
        if self.model is None:
            logger.error("No model loaded. Train or load a model first.")
            return []
            
        try:
            similar_terms = self.model.wv.most_similar(term, topn=n)
            return similar_terms
        except KeyError:
            logger.warning(f"Term '{term}' not in vocabulary")
            return []
    
    def analyze_disease_yield_impact(self):
        """Analyze relationships between diseases and yield impact terms"""
        if self.model is None:
            logger.error("No model loaded. Train or load a model first.")
            return None
        
        # Get flattened list of all disease terms
        disease_terms = []
        for disease, variants in TARGET_DISEASES.items():
            for variant in variants:
                # Add the disease name and its variants
                terms = variant.lower().split()
                if all(term in self.model.wv for term in terms):
                    disease_terms.append((disease, variant, terms))
        
        # Yield impact terms
        yield_terms = ['yield', 'loss', 'reduction', 'damage', 'impact']
        yield_terms = [t for t in yield_terms if t in self.model.wv]
        
        if not disease_terms:
            logger.warning("No disease terms found in vocabulary")
            return None
            
        if not yield_terms:
            logger.warning("No yield terms found in vocabulary")
            return None
        
        # Calculate relationships
        results = []
        
        for disease_key, variant, disease_terms_list in disease_terms:
            # For multi-word terms, average the vectors
            if len(disease_terms_list) > 1:
                try:
                    disease_vector = sum(self.model.wv[word] for word in disease_terms_list) / len(disease_terms_list)
                    
                    # Calculate similarity to yield terms
                    for yield_term in yield_terms:
                        yield_vector = self.model.wv[yield_term]
                        similarity = np.dot(disease_vector, yield_vector) / (
                            np.linalg.norm(disease_vector) * np.linalg.norm(yield_vector))
                        
                        results.append({
                            'disease': disease_key,
                            'variant': variant,
                            'yield_term': yield_term,
                            'similarity': similarity
                        })
                except:
                    logger.warning(f"Could not compute vector for {variant}")
            else:
                # For single-word terms
                term = disease_terms_list[0]
                for yield_term in yield_terms:
                    try:
                        similarity = self.model.wv.similarity(term, yield_term)
                        results.append({
                            'disease': disease_key,
                            'variant': variant,
                            'yield_term': yield_term,
                            'similarity': similarity
                        })
                    except:
                        pass
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if df.empty:
            logger.warning("No disease-yield relationships found")
            return None
            
        # Calculate average similarity by disease
        disease_avg = df.groupby('disease')['similarity'].mean().reset_index()
        disease_avg = disease_avg.sort_values('similarity', ascending=False)
        
        logger.info("Disease yield impact ranking (higher similarity = stronger relationship):")
        for _, row in disease_avg.iterrows():
            logger.info(f"  {row['disease']}: {row['similarity']:.4f}")
            
        return df
    
    def analyze_disease_treatments(self):
        """Analyze relationships between diseases and treatment approaches"""
        if self.model is None:
            logger.error("No model loaded. Train or load a model first.")
            return None
        
        # Get flattened list of all disease terms
        disease_terms = []
        for disease, variants in TARGET_DISEASES.items():
            for variant in variants:
                # Add the disease name and its variants
                terms = variant.lower().split()
                if all(term in self.model.wv for term in terms):
                    disease_terms.append((disease, variant, terms))
        
        # Treatment terms
        treatment_terms = ['fungicide', 'treatment', 'spray', 'application', 'control', 'resistance']
        treatment_terms = [t for t in treatment_terms if t in self.model.wv]
        
        if not disease_terms:
            logger.warning("No disease terms found in vocabulary")
            return None
            
        if not treatment_terms:
            logger.warning("No treatment terms found in vocabulary")
            return None
        
        # Calculate relationships
        results = []
        
        for disease_key, variant, disease_terms_list in disease_terms:
            # For multi-word terms, average the vectors
            if len(disease_terms_list) > 1:
                try:
                    disease_vector = sum(self.model.wv[word] for word in disease_terms_list) / len(disease_terms_list)
                    
                    # Calculate similarity to treatment terms
                    for treatment_term in treatment_terms:
                        treatment_vector = self.model.wv[treatment_term]
                        similarity = np.dot(disease_vector, treatment_vector) / (
                            np.linalg.norm(disease_vector) * np.linalg.norm(treatment_vector))
                        
                        results.append({
                            'disease': disease_key,
                            'variant': variant,
                            'treatment_term': treatment_term,
                            'similarity': similarity
                        })
                except:
                    logger.warning(f"Could not compute vector for {variant}")
            else:
                # For single-word terms
                term = disease_terms_list[0]
                for treatment_term in treatment_terms:
                    try:
                        similarity = self.model.wv.similarity(term, treatment_term)
                        results.append({
                            'disease': disease_key,
                            'variant': variant,
                            'treatment_term': treatment_term,
                            'similarity': similarity
                        })
                    except:
                        pass
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if df.empty:
            logger.warning("No disease-treatment relationships found")
            return None
        
        # Find the most relevant treatment for each disease
        treatment_ranking = df.sort_values(['disease', 'similarity'], ascending=[True, False])
        best_treatments = treatment_ranking.groupby('disease').head(3)
        
        logger.info("Top treatments by disease:")
        for disease in best_treatments['disease'].unique():
            disease_treatments = best_treatments[best_treatments['disease'] == disease]
            treatments = []
            for _, row in disease_treatments.iterrows():
                treatments.append(f"{row['treatment_term']} ({row['similarity']:.4f})")
            logger.info(f"  {disease}: {', '.join(treatments)}")
            
        return df
    
    def visualize_disease_space(self, output_file=None):
        """Create a visualization of the disease-treatment-yield space"""
        if self.model is None:
            logger.error("No model loaded. Train or load a model first.")
            return None
            
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import matplotlib.cm as cm
        except ImportError:
            logger.error("Required libraries missing. Install with: pip install scikit-learn matplotlib")
            return None
        
        # Collect all relevant terms
        terms = []
        vectors = []
        categories = []
        
        # Add disease terms
        for disease, variants in TARGET_DISEASES.items():
            for variant in variants:
                words = variant.split()
                if all(word in self.model.wv for word in words):
                    # Add the combined vector
                    vector = sum(self.model.wv[word] for word in words) / len(words)
                    vectors.append(vector)
                    terms.append(variant)
                    categories.append('disease')
        
        # Add treatment terms
        for term in MANAGEMENT_TERMS:
            words = term.split()
            if all(word in self.model.wv for word in words):
                vector = sum(self.model.wv[word] for word in words) / len(words)
                vectors.append(vector)
                terms.append(term)
                categories.append('treatment')
        
        # Add yield terms
        for term in YIELD_TERMS:
            words = term.split()
            if all(word in self.model.wv for word in words):
                vector = sum(self.model.wv[word] for word in words) / len(words)
                vectors.append(vector)
                terms.append(term)
                categories.append('yield')
                
        # Also add individual key words
        key_words = ['fungicide', 'treatment', 'application', 'yield', 'loss', 'control', 'resistance']
        for word in key_words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
                terms.append(word)
                categories.append('keyword')
                
        if len(vectors) < 3:
            logger.error("Too few terms found in vocabulary for visualization")
            return None
            
        # Convert to array
        vectors = np.array(vectors)
        
        # Apply dimensionality reduction
        if len(vectors) > 50:
            # Use t-SNE for larger datasets
            tsne = TSNE(n_components=2, perplexity=min(30, len(vectors)-1), random_state=42)
            reduced = tsne.fit_transform(vectors)
            method = 't-SNE'
        else:
            # Use PCA for smaller datasets
            pca = PCA(n_components=2, random_state=42)
            reduced = pca.fit_transform(vectors)
            method = 'PCA'
            
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Define colors for categories
        colors = {'disease': 'red', 'treatment': 'blue', 'yield': 'green', 'keyword': 'purple'}
        
        # Plot by category
        for category in colors:
            indices = [i for i, cat in enumerate(categories) if cat == category]
            if indices:
                x = reduced[indices, 0]
                y = reduced[indices, 1]
                plt.scatter(x, y, c=colors[category], label=category, alpha=0.7)
                
                # Add labels
                for i, idx in enumerate(indices):
                    plt.annotate(
                        terms[idx], 
                        (x[i], y[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9
                    )
        
        plt.title(f'Wheat Disease, Treatment, and Yield Term Relationships ({method})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description="Train a specialized wheat disease Word2Vec model")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--vector_size", type=int, default=200, help="Dimensionality of vectors")
    parser.add_argument("--window", type=int, default=7, help="Context window size")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum word frequency")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--visualize", action='store_true', help="Generate visualization")
    parser.add_argument("--output", type=str, default=None, help="Output filename for model")
    
    args = parser.parse_args()
    
    # Validate corpus file
    if not os.path.exists(args.corpus):
        logger.error(f"Corpus file {args.corpus} not found")
        return
        
    # Create and train model
    model = WheatDiseaseModel(args.corpus, args.model_dir)
    model.load_corpus()
    model.preprocess_corpus()
    model.train_model(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers
    )
    
    # Save model
    model_path = model.save_model(args.output)
    
    # Analyze yield impact
    yield_analysis = model.analyze_disease_yield_impact()
    if isinstance(yield_analysis, pd.DataFrame):
        yield_file = os.path.join(args.model_dir, "disease_yield_impact.csv")
        yield_analysis.to_csv(yield_file, index=False)
        logger.info(f"Disease yield impact analysis saved to {yield_file}")
    
    # Analyze treatments
    treatment_analysis = model.analyze_disease_treatments()
    if isinstance(treatment_analysis, pd.DataFrame):
        treatment_file = os.path.join(args.model_dir, "disease_treatments.csv")
        treatment_analysis.to_csv(treatment_file, index=False)
        logger.info(f"Disease treatment analysis saved to {treatment_file}")
    
    # Create visualization
    if args.visualize:
        vis_file = os.path.join(args.model_dir, "disease_space_visualization.png")
        model.visualize_disease_space(output_file=vis_file)
    
    logger.info("Wheat disease model training complete!")

if __name__ == "__main__":
    main()