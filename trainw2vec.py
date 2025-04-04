#!/usr/bin/env python3
"""
Enhanced Word2Vec Training for Wheat Disease Analysis
====================================================

This script implements an optimized approach to training Word2Vec/FastText models
on wheat disease corpus data, with specialized handling of domain terminology.

Features:
- Automatic phrase detection for multi-word terms
- Domain-specific term preservation (yellow_rust, stripe_rust, etc.)
- Enhanced training with seed sentences for key relationships
- Optimized parameters for scientific/agricultural domain
- FastText support for better morphological handling
"""

import os
import re
import json
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import Counter

# Gensim imports
from gensim.models import Word2Vec, FastText
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wheat_word2vec_training.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define wheat disease terms (from your original script)
DISEASE_TERMS = {
    "yellow_rust": ["yellow rust", "stripe rust", "puccinia striiformis", "yr", "pst", "p. striiformis"],
    "brown_rust": ["brown rust", "leaf rust", "puccinia triticina", "puccinia recondita", "lr", "pt", "p. triticina", "p. recondita"],
    "septoria": ["septoria", "septoria tritici", "zymoseptoria tritici", "septoria leaf blotch", 
                "stb", "septoria nodorum", "parastagonospora nodorum", "s. tritici", "z. tritici", "s. nodorum", "p. nodorum"],
    "mildew": ["mildew", "powdery mildew", "blumeria graminis", "erysiphe graminis", "pm", "b. graminis", "e. graminis"]
}

# Define management-related terms
MANAGEMENT_TERMS = [
    "management", "control", "integrated", "strategy", "practice", 
    "rotation", "cultural practice", "prevention", "monitoring", 
    "forecast", "warning system", "disease triangle", "ipm"
]

# Define treatment-specific terms
TREATMENT_TERMS = [
    "treatment", "fungicide", "application", "spray", "timing", "dose", 
    "chemical", "active ingredient", "formulation", "efficacy", "protection", 
    "curative", "preventative", "foliar", "seed treatment", "rate",
    "triazole", "strobilurin", "sdhi", "azole", "tebuconazole", "prothioconazole",
    "epoxiconazole", "pyraclostrobin", "azoxystrobin", "bixafen", "fluxapyroxad"
]

# Define resistance and variety-related terms
RESISTANCE_TERMS = [
    "resistance", "tolerant", "susceptible", "cultivar", "variety", "breeding",
    "gene", "genetic", "immune", "resilient", "sensitive", "germplasm", 
    "genotype", "phenotype", "expression", "marker", "selection", "hybrid",
    "line", "elite", "landrace", "wild relative", "qtl", "genome"
]

# Define epidemiology and weather-related terms
EPIDEMIOLOGY_TERMS = [
    "epidemiology", "lifecycle", "infection", "sporulation", "dispersal", "inoculum",
    "spore", "urediniospore", "teliospore", "aeciospore", "ascospore", "pycnidiospore",
    "conidium", "mycelium", "latent period", "incubation", "temperature", "humidity",
    "moisture", "precipitation", "rainfall", "leaf wetness", "weather", "climate", 
    "winter", "summer", "spring", "autumn", "season", "dew", "wind"
]

# Define symptoms and diagnosis terms
SYMPTOM_TERMS = [
    "symptom", "sign", "lesion", "pustule", "chlorosis", "necrosis", "uredinia",
    "pycnidia", "cleistothecia", "yellow", "brown", "orange", "black", "stripe",
    "spot", "blotch", "patch", "fleck", "discoloration", "diagnostic", "identification",
    "severity", "incidence", "detection", "assessment", "scoring", "grading", "scale"
]

# Financial impact terms
FINANCIAL_IMPACT_TERMS = [
    "financial", "economic", "cost", "expense", "budget", "profit", "loss",
    "revenue", "income", "investment", "return", "yield loss", "million", "billion",
    "dollar", "euro", "pound", "currency", "market", "price", "value", "trade"
]

# Crop terms
CROP_TERMS = [
    "wheat", "triticum aestivum", "triticum durum", "bread wheat", "durum wheat",
    "winter wheat", "spring wheat", "feed wheat", "cereal", "crop", "barley", 
    "oat", "rye", "triticale", "grass", "volunteer", "t. aestivum", "t. durum"
]

# Combine all terms by category for filtering
ALL_CATEGORY_TERMS = {
    "disease": [term for terms in DISEASE_TERMS.values() for term in terms],
    "management": MANAGEMENT_TERMS,
    "treatment": TREATMENT_TERMS,
    "resistance": RESISTANCE_TERMS,
    "financial": FINANCIAL_IMPACT_TERMS,
    "epidemiology": EPIDEMIOLOGY_TERMS,
    "symptom": SYMPTOM_TERMS,
    "crop": CROP_TERMS
}

# Epoch logger callback
class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress after each epoch"""
    def __init__(self):
        self.epoch = 0
        self.start_time = None
        
    def on_epoch_begin(self, model):
        import time
        self.start_time = time.time()
        logger.info(f"Epoch #{self.epoch + 1} starting")
        
    def on_epoch_end(self, model):
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"Epoch #{self.epoch + 1} completed in {elapsed:.1f} seconds")
        self.epoch += 1

def create_domain_term_mapping():
    """Create a mapping from multi-word domain terms to underscore versions"""
    domain_terms = {}
    
    # Process disease terms
    for disease, terms in DISEASE_TERMS.items():
        # Add the disease key itself if it contains spaces
        if ' ' in disease:
            domain_terms[disease] = disease.replace(' ', '_')
            
        # Add all the terms for this disease
        for term in terms:
            if ' ' in term:
                domain_terms[term] = term.replace(' ', '_')
    
    # Process all other category terms
    for category, terms in ALL_CATEGORY_TERMS.items():
        for term in terms:
            if ' ' in term:
                domain_terms[term] = term.replace(' ', '_')
    
    logger.info(f"Created mapping for {len(domain_terms)} domain-specific terms")
    return domain_terms

def generate_seed_sentences():
    """Generate seed sentences to enhance training for key relationships"""
    seed_sentences = []
    
    # Disease-symptom relationships
    for disease in DISEASE_TERMS:
        disease_term = disease.replace(' ', '_')
        for symptom in SYMPTOM_TERMS[:10]:  # Use top symptoms
            symptom_term = symptom.replace(' ', '_')
            seed_sentences.append([disease_term, "causes", symptom_term])
            seed_sentences.append([symptom_term, "indicates", disease_term])
    
    # Disease-management relationships
    for disease in DISEASE_TERMS:
        disease_term = disease.replace(' ', '_')
        for mgmt in MANAGEMENT_TERMS[:8]:
            mgmt_term = mgmt.replace(' ', '_')
            seed_sentences.append([disease_term, "controlled_by", mgmt_term])
            seed_sentences.append([mgmt_term, "controls", disease_term])
    
    # Disease-treatment relationships
    for disease in DISEASE_TERMS:
        disease_term = disease.replace(' ', '_')
        for treatment in TREATMENT_TERMS[:8]:
            treatment_term = treatment.replace(' ', '_')
            seed_sentences.append([disease_term, "treated_with", treatment_term])
            seed_sentences.append([treatment_term, "effective_against", disease_term])
    
    # Disease-environment relationships
    for disease in DISEASE_TERMS:
        disease_term = disease.replace(' ', '_')
        for env in EPIDEMIOLOGY_TERMS[:8]:
            env_term = env.replace(' ', '_')
            seed_sentences.append([disease_term, "favored_by", env_term])
            seed_sentences.append([env_term, "promotes", disease_term])
    
    logger.info(f"Generated {len(seed_sentences)} seed sentences for domain relationships")
    return seed_sentences

def preprocess_corpus(corpus_file, domain_terms):
    """
    Preprocess corpus with specialized handling for domain terms
    
    Args:
        corpus_file: Path to corpus text file
        domain_terms: Mapping of domain terms to their underscore versions
        
    Returns:
        List of tokenized sentences
    """
    logger.info(f"Reading corpus from {corpus_file}")
    
    # Read corpus
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(lines)} lines from corpus")
    except Exception as e:
        logger.error(f"Error reading corpus: {e}")
        return []
    
    # First pass: Replace domain terms with underscore versions
    logger.info("Preprocessing corpus lines with domain term replacement")
    processed_lines = []
    
    # Sort terms by length (longest first) to avoid partial replacements
    sorted_terms = sorted(domain_terms.items(), key=lambda x: len(x[0]), reverse=True)
    
    # Track replacements for logging
    replacement_counts = Counter()
    
    for line in tqdm(lines, desc="Replacing domain terms"):
        line_lower = line.lower()
        original_line = line_lower
        
        for term, replacement in sorted_terms:
            # Use word boundary pattern to avoid partial word matches
            pattern = r'\b' + re.escape(term) + r'\b'
            new_line = re.sub(pattern, replacement, line_lower)
            
            # Count replacements
            if new_line != line_lower:
                count = len(re.findall(pattern, line_lower))
                replacement_counts[term] += count
                line_lower = new_line
        
        processed_lines.append(line_lower)
        
    # Log replacement statistics
    logger.info(f"Term replacement statistics:")
    for term, count in replacement_counts.most_common(20):
        logger.info(f"  '{term}' → '{domain_terms[term]}': {count} replacements")
    
    # Second pass: Detect common phrases
    logger.info("Detecting common phrases in corpus")
    
    # First detect common bigrams
    phrases = Phrases(
        [simple_preprocess(line) for line in processed_lines], 
        min_count=5, 
        threshold=10.0,
        connector_words=ENGLISH_CONNECTOR_WORDS
    )
    bigram_model = phrases.freeze()
    
    # Then detect trigrams from bigrams
    logger.info("Detecting trigrams from bigrams")
    phrases_trigram = Phrases(
        [bigram_model[simple_preprocess(line)] for line in processed_lines],
        min_count=3,
        threshold=5.0,
        connector_words=ENGLISH_CONNECTOR_WORDS
    )
    trigram_model = phrases_trigram.freeze()
    
    # Third pass: Tokenize and apply phrase models
    logger.info("Tokenizing and applying phrase models")
    sentences = []
    
    for line in tqdm(processed_lines, desc="Tokenizing sentences"):
        tokens = simple_preprocess(line)
        tokens = trigram_model[bigram_model[tokens]]
        
        if tokens and len(tokens) >= 3:  # Filter very short sentences
            sentences.append(tokens)
    
    logger.info(f"Created {len(sentences)} tokenized sentences")
    return sentences

def train_enhanced_word2vec(
    corpus_file, 
    output_dir,
    model_type='word2vec',  # 'word2vec' or 'fasttext'
    vector_size=300, 
    window=8, 
    min_count=5, 
    epochs=20,
    seed_repetitions=25
):
    """
    Train an enhanced Word2Vec or FastText model with domain-specific optimizations
    
    Args:
        corpus_file: Path to corpus text file
        output_dir: Directory to save the model and reports
        model_type: 'word2vec' or 'fasttext'
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        epochs: Number of training iterations
        seed_repetitions: Number of times to repeat seed sentences
        
    Returns:
        Trained model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize callback
    epoch_logger = EpochLogger()
    
    # Phase 1: Create domain term mapping
    domain_terms = create_domain_term_mapping()
    
    # Phase 2: Generate seed sentences for key relationships
    seed_sentences = generate_seed_sentences()
    
    # Phase 3: Preprocess corpus
    sentences = preprocess_corpus(corpus_file, domain_terms)
    
    # Phase 4: Add seed sentences
    logger.info(f"Adding seed sentences (x{seed_repetitions}) to training data")
    for _ in range(seed_repetitions):
        sentences.extend(seed_sentences)
    
    # Phase 5: Train the model
    logger.info(f"Training {model_type} model on {len(sentences)} sentences")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")
    
    if model_type.lower() == 'fasttext':
        # Use FastText for better morphological handling
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=8,
            epochs=epochs,
            sg=1,              # Skip-gram
            negative=15,       # Negative sampling
            callbacks=[epoch_logger]
        )
        model_path = os.path.join(output_dir, "wheat_disease_fasttext.model")
    else:
        # Use standard Word2Vec
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=8,
            epochs=epochs,
            sg=1,              # Skip-gram
            negative=15,       # Negative sampling
            callbacks=[epoch_logger]
        )
        model_path = os.path.join(output_dir, "wheat_disease_word2vec.model")
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test the model
    test_model(model, output_dir)
    
    return model

def test_model(model, output_dir):
    """
    Test the trained model and generate a report
    
    Args:
        model: Trained Word2Vec or FastText model
        output_dir: Directory to save reports
    """
    # Test words including compound terms
    test_words = [
        # Disease terms with underscores
        'yellow_rust', 'stripe_rust', 'brown_rust', 'leaf_rust', 
        'septoria', 'septoria_tritici', 'powdery_mildew',
        # Management terms
        'fungicide', 'resistance', 'application', 'spray', 'timing',
        # Environmental terms
        'temperature', 'humidity', 'rainfall', 'climate',
        # Symptom terms
        'pustule', 'lesion', 'stripe', 'blotch',
        # Impact terms
        'yield', 'loss', 'economic'
    ]
    
    # Initialize report
    report = {
        "model_info": {
            "type": "FastText" if isinstance(model, FastText) else "Word2Vec",
            "vector_size": model.vector_size,
            "window": model.window,
            "min_count": model.min_count,
            "epochs": model.epochs,
            "vocab_size": len(model.wv.index_to_key),
            "test_date": datetime.now().isoformat()
        },
        "term_coverage": {
            "tested": len(test_words),
            "found": 0,
            "coverage_percent": 0
        },
        "similar_terms": {}
    }
    
    # Test each word
    found_words = 0
    logger.info("\nTesting model with key domain terms:")
    
    for word in test_words:
        if word in model.wv:
            found_words += 1
            logger.info(f"\nWords similar to '{word}':")
            
            similar_words = model.wv.most_similar(word, topn=10)
            report["similar_terms"][word] = [
                {"term": similar_word, "similarity": float(similarity)} 
                for similar_word, similarity in similar_words
            ]
            
            # Log top 5 for console output
            for similar_word, similarity in similar_words[:5]:
                logger.info(f"  {similar_word}: {similarity:.4f}")
        else:
            logger.info(f"\nWord '{word}' not in vocabulary")
    
    # Update report with coverage stats
    coverage_percent = found_words / len(test_words) * 100
    report["term_coverage"]["found"] = found_words
    report["term_coverage"]["coverage_percent"] = coverage_percent
    
    # Print summary
    logger.info(f"\nTest Summary:")
    logger.info(f"Vocabulary size: {len(model.wv.index_to_key)} words")
    logger.info(f"Found {found_words}/{len(test_words)} test words in vocabulary ({coverage_percent:.1f}%)")
    
    # Save report
    report_path = os.path.join(output_dir, "model_test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Test report saved to {report_path}")
    
    # Test specific relationships
    test_relationships(model, output_dir)

def test_relationships(model, output_dir):
    """
    Test specific domain relationships in the model
    
    Args:
        model: Trained Word2Vec or FastText model
        output_dir: Directory to save reports
    """
    # Define relationship tests with both forms of terms
    relationship_tests = [
        # Disease-symptom relationships
        {"category": "disease_symptom", "a": "yellow_rust", "rel": "causes", "b": "stripe"},
        {"category": "disease_symptom", "a": "brown_rust", "rel": "causes", "b": "pustule"},
        {"category": "disease_symptom", "a": "septoria", "rel": "causes", "b": "blotch"},
        {"category": "disease_symptom", "a": "powdery_mildew", "rel": "causes", "b": "mycelium"},
        
        # Disease-management relationships
        {"category": "disease_management", "a": "yellow_rust", "rel": "controlled_by", "b": "fungicide"},
        {"category": "disease_management", "a": "brown_rust", "rel": "controlled_by", "b": "spray"},
        {"category": "disease_management", "a": "septoria", "rel": "controlled_by", "b": "application"},
        {"category": "disease_management", "a": "powdery_mildew", "rel": "controlled_by", "b": "treatment"},
        
        # Disease-environment relationships
        {"category": "disease_environment", "a": "yellow_rust", "rel": "favored_by", "b": "temperature"},
        {"category": "disease_environment", "a": "brown_rust", "rel": "favored_by", "b": "moisture"},
        {"category": "disease_environment", "a": "septoria", "rel": "favored_by", "b": "rainfall"},
        {"category": "disease_environment", "a": "powdery_mildew", "rel": "favored_by", "b": "humidity"}
    ]
    
    # Initialize results
    relationship_results = []
    
    # Test each relationship
    logger.info("\nTesting domain-specific relationships:")
    
    for test in relationship_tests:
        result = {
            "category": test["category"],
            "relationship": f"{test['a']} {test['rel']} {test['b']}",
            "valid": False,
            "details": {}
        }
        
        # Try both forms of terms (with underscore and with space)
        a_terms = [test["a"], test["a"].replace('_', ' ')]
        b_terms = [test["b"], test["b"].replace('_', ' ')]
        
        # Find which forms exist in the vocabulary
        a_in_vocab = next((term for term in a_terms if term in model.wv), None)
        b_in_vocab = next((term for term in b_terms if term in model.wv), None)
        
        # Check if terms are in vocabulary
        if a_in_vocab and b_in_vocab:
            # Calculate similarity
            similarity = model.wv.similarity(a_in_vocab, b_in_vocab)
            result["details"]["direct_similarity"] = float(similarity)
            result["details"]["terms_used"] = {"a": a_in_vocab, "b": b_in_vocab}
            
            # Try analogy test
            try:
                # This tests if a is to b as c is to ?
                similar_terms = model.wv.most_similar(
                    positive=[a_in_vocab, test["rel"]], 
                    topn=10
                )
                
                # Check if expected term is in results - check both forms
                for i, (term, score) in enumerate(similar_terms):
                    if term in b_terms or any(term.startswith(b) for b in b_terms):
                        result["valid"] = True
                        result["details"]["rank"] = i + 1
                        result["details"]["score"] = float(score)
                        break
                
                result["details"]["top_terms"] = [
                    {"term": term, "score": float(score)} 
                    for term, score in similar_terms[:5]
                ]
            except Exception as e:
                result["details"]["error"] = str(e)
        else:
            missing = []
            if not a_in_vocab:
                missing.append(test["a"])
            if not b_in_vocab:
                missing.append(test["b"])
            if test["rel"] not in model.wv:
                missing.append(test["rel"])
            result["details"]["missing_terms"] = missing
        
        # Log result
        status = "✓" if result["valid"] else "✗"
        logger.info(f"{status} {test['a']} {test['rel']} {test['b']}")
        if "top_terms" in result["details"]:
            terms = [t["term"] for t in result["details"]["top_terms"]]
            logger.info(f"  Top terms: {', '.join(terms)}")
        
        relationship_results.append(result)
def main():
    parser = argparse.ArgumentParser(description="Enhanced Word2Vec Training for Wheat Disease Analysis")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus text file")
    parser.add_argument("--output_dir", type=str, default="wheat_vectors", help="Output directory")
    parser.add_argument("--model_type", type=str, choices=['word2vec', 'fasttext'], default='fasttext', 
                        help="Type of model to train")
    parser.add_argument("--vector_size", type=int, default=300, help="Dimension of word vectors")
    parser.add_argument("--window", type=int, default=8, help="Context window size")
    parser.add_argument("--min_count", type=int, default=5, help="Minimum word frequency")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training iterations")
    parser.add_argument("--seed_repetitions", type=int, default=25,  # Increased from 15 to 25
                        help="Number of times to repeat seed sentences")
    
    args = parser.parse_args()
    
    logger.info("Starting enhanced Word2Vec training for wheat disease analysis")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model type: {args.model_type}")
    
    # Train model
    train_enhanced_word2vec(
        corpus_file=args.corpus,
        output_dir=args.output_dir,
        model_type=args.model_type,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        seed_repetitions=args.seed_repetitions
    )
    
    logger.info("Training complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)