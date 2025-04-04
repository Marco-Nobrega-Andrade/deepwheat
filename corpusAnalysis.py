#!/usr/bin/env python3
"""
Wheat Disease Corpus Analysis Tool
=================================

This script analyzes a corpus file for wheat disease research to:
1. Check how disease terms appear (with spaces or underscores)
2. Count frequency of domain-specific terms
3. Examine context of key terms
4. Generate statistics about term usage
5. Identify potential preprocessing issues
"""

import os
import re
import json
import argparse
import random
from collections import Counter, defaultdict
from datetime import datetime

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

# Define symptoms and diagnosis terms
SYMPTOM_TERMS = [
    "symptom", "sign", "lesion", "pustule", "chlorosis", "necrosis", "uredinia",
    "pycnidia", "cleistothecia", "yellow", "brown", "orange", "black", "stripe",
    "spot", "blotch", "patch", "fleck", "discoloration", "diagnostic", "identification",
    "severity", "incidence", "detection", "assessment", "scoring", "grading", "scale"
]

# Define epidemiology and weather-related terms
EPIDEMIOLOGY_TERMS = [
    "epidemiology", "lifecycle", "infection", "sporulation", "dispersal", "inoculum",
    "spore", "urediniospore", "teliospore", "aeciospore", "ascospore", "pycnidiospore",
    "conidium", "mycelium", "latent period", "incubation", "temperature", "humidity",
    "moisture", "precipitation", "rainfall", "leaf wetness", "weather", "climate", 
    "winter", "summer", "spring", "autumn", "season", "dew", "wind"
]

# Combine all terms by category for analysis
ALL_CATEGORY_TERMS = {
    "disease": [term for disease, terms in DISEASE_TERMS.items() for term in terms] + list(DISEASE_TERMS.keys()),
    "management": MANAGEMENT_TERMS,
    "treatment": TREATMENT_TERMS,
    "symptom": SYMPTOM_TERMS,
    "epidemiology": EPIDEMIOLOGY_TERMS
}

def analyze_term_frequencies(corpus_file, max_sample_contexts=5):
    """
    Analyze term frequencies and formats (with spaces or underscores) in the corpus
    
    Args:
        corpus_file: Path to the corpus file
        max_sample_contexts: Maximum number of sample contexts to collect per term
        
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing term frequencies in {corpus_file}...")
    
    # Read the corpus
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines from corpus")
    except Exception as e:
        print(f"Error reading corpus: {e}")
        return {}
    
    # Initialize results
    results = {
        "corpus_info": {
            "filename": os.path.basename(corpus_file),
            "line_count": len(lines),
            "analysis_date": datetime.now().isoformat()
        },
        "term_counts": {},
        "category_summary": {},
        "sample_contexts": {}
    }
    
    # Create flat list of all terms to analyze
    all_terms = []
    for category, terms in ALL_CATEGORY_TERMS.items():
        for term in terms:
            all_terms.append((term, category))
    
    # Main diseases separately for clarity
    for disease in DISEASE_TERMS.keys():
        all_terms.append((disease, "disease"))
    
    # Also add underscore versions
    underscore_terms = []
    for term, category in all_terms:
        if ' ' in term:
            underscore_terms.append((term.replace(' ', '_'), category))
    all_terms.extend(underscore_terms)
    
    # Count term frequencies
    term_counts = {}
    term_contexts = defaultdict(list)
    
    # Process each line
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        for term, category in all_terms:
            # Use word boundary pattern to find whole terms
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, line_lower):
                # Count occurrence
                term_counts[term] = term_counts.get(term, 0) + 1
                
                # Collect sample context (up to max_sample_contexts per term)
                if len(term_contexts[term]) < max_sample_contexts:
                    # Get context (up to 50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(line_lower), match.end() + 50)
                    
                    context = line_lower[start:end]
                    # Highlight the term in context
                    context = context.replace(term, f"**{term}**")
                    
                    term_contexts[term].append({
                        "line_number": i + 1,
                        "context": context
                    })
    
    # Organize term counts by original term (with space) and underscore version
    organized_counts = {}
    for term, category in all_terms:
        base_term = term.replace('_', ' ')
        underscore_term = term.replace(' ', '_')
        
        if base_term not in organized_counts:
            organized_counts[base_term] = {
                "category": category,
                "with_spaces": term_counts.get(base_term, 0),
                "with_underscores": term_counts.get(underscore_term, 0)
            }
    
    # Calculate category summaries
    category_summary = {}
    for category, terms in ALL_CATEGORY_TERMS.items():
        category_terms = [(term, term.replace(' ', '_')) for term in terms]
        
        # Count occurrences by category
        spaces_count = sum(term_counts.get(term_with_spaces, 0) for term_with_spaces, _ in category_terms)
        underscore_count = sum(term_counts.get(term_with_underscores, 0) for _, term_with_underscores in category_terms)
        
        category_summary[category] = {
            "total_terms": len(terms),
            "terms_found_with_spaces": sum(1 for term_with_spaces, _ in category_terms if term_counts.get(term_with_spaces, 0) > 0),
            "terms_found_with_underscores": sum(1 for _, term_with_underscores in category_terms if term_counts.get(term_with_underscores, 0) > 0),
            "occurrences_with_spaces": spaces_count,
            "occurrences_with_underscores": underscore_count,
            "total_occurrences": spaces_count + underscore_count
        }
    
    # Add results
    results["term_counts"] = organized_counts
    results["category_summary"] = category_summary
    results["sample_contexts"] = term_contexts
    
    return results