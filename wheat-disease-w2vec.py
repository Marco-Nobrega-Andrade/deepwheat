#!/usr/bin/env python3
"""
Enhanced Wheat Disease Text Extraction for NLP Analysis
=======================================================

This script extracts text from PDF documents about wheat diseases,
handles tables, figures, and creates a clean corpus suitable for 
Word2Vec training and other NLP analyses.

Features:
- Improved table detection and handling
- Structure detection (headings, sections, etc.)
- Better extraction of numerical data
- More sophisticated sentence filtering
- Disease-specific corpus creation
- Better metadata extraction
"""

import os
import re
import sys
import fitz  # PyMuPDF
import argparse
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import tabula
import logging
import json
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_extraction.log", mode='w'),
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

# Define wheat disease terms for filtering relevant content
DISEASE_TERMS = {
    "yellow_rust": ["yellow rust", "stripe rust", "puccinia striiformis", "yr", "pst", "p. striiformis"],
    "brown_rust": ["brown rust", "leaf rust", "puccinia triticina", "puccinia recondita", "lr", "pt", "p. triticina", "p. recondita"],
    "septoria": ["septoria", "septoria tritici", "zymoseptoria tritici", "septoria leaf blotch", 
                "stb", "septoria nodorum", "parastagonospora nodorum", "s. tritici", "z. tritici", "s. nodorum", "p. nodorum"],
    "mildew": ["mildew", "powdery mildew", "blumeria graminis", "erysiphe graminis", "pm", "b. graminis", "e. graminis"]
}

# Define management-related terms for filtering relevant content
MANAGEMENT_TERMS = [
    "management", "control", "integrated", "strategy", "practice", 
    "rotation", "cultural practice", "prevention", "monitoring", 
    "forecast", "warning system", "disease triangle", "ipm"
]

# Define treatment-specific terms for filtering relevant content
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

# Define financial impact terms for filtering relevant content
FINANCIAL_IMPACT_TERMS = [
    "financial", "economic", "cost", "expense", "budget", "profit", "loss",
    "revenue", "income", "investment", "return", "yield loss", "million", "billion",
    "dollar", "euro", "pound", "currency", "market", "price", "value", "trade",
    "export", "import", "gdp", "economy", "subsidy", "insurance", "compensation"
]

# Define epidemiology and weather-related terms
EPIDEMIOLOGY_TERMS = [
    "epidemiology", "lifecycle", "infection", "sporulation", "dispersal", "inoculum",
    "spore", "urediniospore", "teliospore", "aeciospore", "ascospore", "pycnidiospore",
    "conidium", "mycelium", "latent period", "incubation", "temperature", "humidity",
    "moisture", "precipitation", "rainfall", "leaf wetness", "weather", "climate", 
    "winter", "summer", "spring", "autumn", "season", "dew", "wind"
]

# Define specific crops and host plants
CROP_TERMS = [
    "wheat", "triticum aestivum", "triticum durum", "bread wheat", "durum wheat",
    "winter wheat", "spring wheat", "feed wheat", "cereal", "crop", "barley", 
    "oat", "rye", "triticale", "grass", "volunteer", "t. aestivum", "t. durum"
]

# Define symptoms and diagnosis terms
SYMPTOM_TERMS = [
    "symptom", "sign", "lesion", "pustule", "chlorosis", "necrosis", "uredinia",
    "pycnidia", "cleistothecia", "yellow", "brown", "orange", "black", "stripe",
    "spot", "blotch", "patch", "fleck", "discoloration", "diagnostic", "identification",
    "severity", "incidence", "detection", "assessment", "scoring", "grading", "scale"
]

# Combine all terms by category for filtering
ALL_CATEGORY_TERMS = {
    "disease": [term for terms in DISEASE_TERMS.values() for term in terms],
    "management": MANAGEMENT_TERMS,
    "treatment": TREATMENT_TERMS,
    "resistance": RESISTANCE_TERMS,
    "financial": FINANCIAL_IMPACT_TERMS,
    "epidemiology": EPIDEMIOLOGY_TERMS,
    "crop": CROP_TERMS,
    "symptom": SYMPTOM_TERMS
}

# Flatten all terms for general filtering
ALL_TERMS = [term for terms in ALL_CATEGORY_TERMS.values() for term in terms]

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None
        self.text = ""
        self.tables = []
        self.metadata = {}
        self.structure = {}
        
    def open(self):
        """Open the PDF document"""
        try:
            self.doc = fitz.open(self.pdf_path)
            return True
        except Exception as e:
            logger.error(f"Error opening {self.pdf_path}: {e}")
            return False
    
    def extract_metadata(self):
        """Extract metadata from the PDF"""
        if not self.doc:
            return {}
        
        meta = self.doc.metadata
        if meta:
            self.metadata = {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "keywords": meta.get("keywords", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "creation_date": meta.get("creationDate", ""),
                "modification_date": meta.get("modDate", ""),
                "num_pages": len(self.doc),
                "file_size": os.path.getsize(self.pdf_path)
            }
        
        # Try to extract more accurate title if metadata title is missing
        if not self.metadata.get("title"):
            # Get text from first page
            if len(self.doc) > 0:
                first_page_text = self.doc[0].get_text()
                lines = first_page_text.split('\n')
                
                # Look for potential title in first few lines
                potential_title = ""
                for line in lines[:10]:  # Consider first 10 lines
                    if line and len(line) > 10 and len(line) < 200:  # Reasonable title length
                        if not re.search(r'\d+/\d+/\d+', line):  # Skip date lines
                            potential_title = line
                            break
                
                if potential_title:
                    self.metadata["potential_title"] = potential_title
        
        return self.metadata
    
    def extract_text_content(self):
        """Extract text content from all pages"""
        if not self.doc:
            return ""
        
        all_text = []
        self.structure = {"sections": [], "num_pages": len(self.doc)}
        
        for page_idx, page in enumerate(self.doc):
            # Extract text from page
            page_text = page.get_text()
            all_text.append(page_text)
            
            # Try to identify section headers
            lines = page_text.split('\n')
            for line_idx, line in enumerate(lines):
                if self._is_likely_header(line):
                    self.structure.setdefault("headers", []).append({
                        "page": page_idx + 1,
                        "text": line,
                        "position": line_idx
                    })
        
        self.text = "\n".join(all_text)
        return self.text
    
    def _is_likely_header(self, line):
        """Check if a line is likely a section header"""
        line = line.strip()
        if not line:
            return False
        
        # Check for common section header patterns
        if re.match(r'^[0-9]+\.\s+[A-Z]', line):  # Numbered section like "1. INTRODUCTION"
            return True
        if re.match(r'^[A-Z][A-Za-z\s]{2,30}$', line) and len(line) < 50:  # All caps or title case short phrase
            return True
        if re.match(r'^(Introduction|Materials and Methods|Results|Discussion|Conclusion|References|Abstract)$', line, re.IGNORECASE):
            return True
        
        return False
    
    def extract_tables(self):
        """Extract tables from the PDF using tabula-py"""
        # First try using PyMuPDF to identify table regions
        self._extract_tables_with_pymupdf()
        
        # Then try tabula-py for more structured table extraction
        try:
            # This requires Java to be installed
            tables = tabula.read_pdf(self.pdf_path, pages='all', multiple_tables=True)
            if tables:
                for i, table in enumerate(tables):
                    if not table.empty:
                        table_dict = table.to_dict('records')
                        if table_dict:  # Make sure there's actual content
                            self.tables.append({
                                "extraction_method": "tabula",
                                "table_index": i,
                                "rows": len(table),
                                "columns": len(table.columns),
                                "data": table_dict
                            })
        except Exception as e:
            logger.warning(f"Tabula table extraction failed for {self.pdf_path}: {e}")
        
        return self.tables
    
    def _extract_tables_with_pymupdf(self):
        """Use PyMuPDF to extract tables"""
        if not self.doc:
            return []
        
        for page_idx, page in enumerate(self.doc):
            try:
                # Find tables on the page
                tables = page.find_tables()
                if tables and tables.tables:
                    for table_idx, table in enumerate(tables.tables):
                        # Extract as a list of lists
                        table_data = []
                        for row_idx in range(table.row_count):
                            row_data = []
                            for col_idx in range(table.col_count):
                                cell_text = table.cells[row_idx][col_idx].text if table.cells[row_idx][col_idx] else ""
                                row_data.append(cell_text)
                            if any(cell for cell in row_data):  # Skip empty rows
                                table_data.append(row_data)
                        
                        if table_data:  # Only add if we have data
                            self.tables.append({
                                "extraction_method": "pymupdf",
                                "page": page_idx + 1,
                                "table_index": table_idx,
                                "rows": len(table_data),
                                "columns": len(table_data[0]) if table_data else 0,
                                "data": table_data
                            })
            except Exception as e:
                logger.warning(f"PyMuPDF table extraction failed for page {page_idx+1}: {e}")
    
    def clean_text(self):
        """Clean the extracted text to remove PDF artifacts and formatting"""
        if not self.text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', self.text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove page numbers
        text = re.sub(r'\b\d+\s*\/\s*\d+\b', '', text)
        
        # Remove timestamp-like patterns
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}', '', text)
        
        # Fix line breaks that split words
        text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
        
        # Remove repeated header/footer text that appears on multiple pages
        if self.structure.get("headers"):
            for header in self.structure["headers"]:
                # Only remove headers that appear multiple times
                if header["text"] and len(header["text"]) > 10:
                    text = text.replace(header["text"], "", text.count(header["text"]) - 1)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        self.text = text.strip()
        return self.text
    
    def identify_disease_focus(self):
        """Identify which disease(s) this document focuses on"""
        if not self.text:
            return []
        
        text_lower = self.text.lower()
        disease_counts = {}
        
        for disease, terms in DISEASE_TERMS.items():
            count = sum(text_lower.count(term) for term in terms)
            disease_counts[disease] = count
        
        # Determine main focus - disease(s) that appear significantly
        total_mentions = sum(disease_counts.values())
        if total_mentions == 0:
            return []
        
        focus_diseases = []
        for disease, count in disease_counts.items():
            # Consider it a focus if it represents at least 15% of all disease mentions
            # or has at least 5 mentions
            if count >= 5 or (total_mentions > 0 and count / total_mentions >= 0.15):
                focus_diseases.append(disease)
        
        return focus_diseases
    
    def extract_and_process(self):
        """Complete extraction and processing pipeline"""
        if not self.open():
            return False
        
        # Extract metadata
        self.extract_metadata()
        
        # Extract text
        self.extract_text_content()
        
        # Extract tables
        self.extract_tables()
        
        # Clean text
        self.clean_text()
        
        # Identify disease focus
        disease_focus = self.identify_disease_focus()
        self.metadata["disease_focus"] = disease_focus
        
        # Create a document summary
        self.summary = {
            "filename": os.path.basename(self.pdf_path),
            "path": self.pdf_path,
            "metadata": self.metadata,
            "disease_focus": disease_focus,
            "num_pages": self.structure.get("num_pages", 0),
            "num_tables": len(self.tables),
            "text_length": len(self.text),
            "processing_time": datetime.now().isoformat()
        }
        
        # Close the document
        if self.doc:
            self.doc.close()
            self.doc = None
        
        return True
    
    def get_sentences(self):
        """Split the cleaned text into sentences"""
        if not self.text:
            return []
        
        # First split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', self.text)
        
        sentences = []
        for paragraph in paragraphs:
            # Skip very short paragraphs likely to be noise
            if len(paragraph) < 40:
                continue
                
            # Use NLTK for sentence tokenization
            try:
                para_sentences = sent_tokenize(paragraph)
                sentences.extend([s.strip() for s in para_sentences if len(s.strip()) > 10])
            except Exception as e:
                # Fallback to simple splitting if NLTK fails
                simple_sentences = re.split(r'[.!?]+', paragraph)
                sentences.extend([s.strip() for s in simple_sentences if len(s.strip()) > 10])
        
        return sentences
    
    def get_relevant_sentences(self):
        """Get sentences relevant to wheat disease management"""
        sentences = self.get_sentences()
        
        relevant_sentences = []
        sentence_categories = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip if likely table text
            if self._is_likely_table_text(sentence):
                continue
                
            # Skip very short sentences
            if len(sentence.split()) < 5 or len(sentence) < 40:
                continue
            
            # Skip likely headers, page numbers, URLs, etc.
            if sentence.isupper() or re.match(r'^(Figure|Table|Page|Chapter)\s+\d+', sentence):
                continue
            
            # Track which categories this sentence matches
            matching_categories = []
            
            # Check all categories
            for category, terms in ALL_CATEGORY_TERMS.items():
                if any(term in sentence_lower for term in terms):
                    matching_categories.append(category)
            
            # Keep sentences that match disease plus another category, or match multiple categories
            if ('disease' in matching_categories and len(matching_categories) > 1) or len(matching_categories) > 2:
                relevant_sentences.append(sentence)
                sentence_categories.append(matching_categories)
        
        return relevant_sentences, sentence_categories
    
    def _is_likely_table_text(self, text):
        """Detect text that's likely from poorly extracted tables"""
        # Check for patterns typical of poorly extracted tables
        if re.search(r'^\s*[\d\.]+\s*$', text):  # Just a number
            return True
        if re.search(r'^\s*[a-zA-Z0-9]+\s*$', text):  # Single word or code
            return True
        if text.count('\t') > 3:  # Lots of tabs
            return True
        if re.search(r'\d+(\s+\d+){3,}', text):  # Multiple numbers in sequence
            return True
        if re.search(r'^\s*\w+(\s+\|+\s+\w+)+\s*$', text):  # Words separated by vertical bars
            return True
        # Check for patterns of repeated whitespace which often indicate table cells
        if re.search(r'(\s{3,}[^\s]+){3,}', text):
            return True
        return False
    
    def get_structured_tables(self):
        """Convert raw table data to structured format suitable for analysis"""
        structured_tables = []
        
        for table in self.tables:
            # Skip tables with no data
            if not table.get("data"):
                continue
                
            # Different process based on extraction method
            if table["extraction_method"] == "tabula":
                # Already structured as dictionaries
                structured_tables.append({
                    "table_index": table["table_index"],
                    "extraction_method": "tabula",
                    "rows": table["rows"],
                    "columns": table["columns"],
                    "data": table["data"]
                })
            elif table["extraction_method"] == "pymupdf":
                # Convert to dict format
                # First row is likely header
                header = table["data"][0] if table["data"] else []
                data = []
                
                if table["data"] and len(table["data"]) > 1:
                    for row in table["data"][1:]:  # Skip header row
                        row_dict = {}
                        for i, value in enumerate(row):
                            key = header[i] if i < len(header) else f"column_{i}"
                            row_dict[key] = value
                        data.append(row_dict)
                
                structured_tables.append({
                    "table_index": table["table_index"],
                    "page": table.get("page", 0),
                    "extraction_method": "pymupdf",
                    "rows": len(data),
                    "columns": len(header),
                    "headers": header,
                    "data": data
                })
        
        return structured_tables
    
    def close(self):
        """Ensure document is closed"""
        if self.doc:
            self.doc.close()
            self.doc = None

def process_pdf_file(pdf_path, output_dir=None):
    """Process a single PDF file and extract content"""
    processor = PDFProcessor(pdf_path)
    
    logger.info(f"Processing {pdf_path}")
    success = processor.extract_and_process()
    
    if not success:
        logger.error(f"Failed to process {pdf_path}")
        return None
    
    # Get relevant sentences
    relevant_sentences, categories = processor.get_relevant_sentences()
    
    # Get tables
    tables = processor.get_structured_tables()
    
    result = {
        "summary": processor.summary,
        "sentences": relevant_sentences,
        "sentence_categories": categories if len(categories) == len(relevant_sentences) else None,
        "tables": tables
    }
    
    # Write results to file if output_dir specified
    if output_dir:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Write sentences to text file
        sentences_file = os.path.join(output_dir, f"{base_name}_sentences.txt")
        with open(sentences_file, 'w', encoding='utf-8') as f:
            for sentence in relevant_sentences:
                f.write(sentence + '\n')
        
        # Write full result to JSON
        result_file = os.path.join(output_dir, f"{base_name}_data.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Clean up
    processor.close()
    
    return result

def process_pdf_directory(pdf_dir, output_dir, filter_disease=None):
    """
    Process all PDFs in a directory and create a corpus for analysis
    
    Args:
        pdf_dir: Directory containing PDFs
        output_dir: Directory to store output files
        filter_disease: Optional filter to process only PDFs for a specific disease
    
    Returns:
        Summary of processing results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create disease-specific output directories
    for disease in DISEASE_TERMS:
        disease_dir = os.path.join(output_dir, disease)
        os.makedirs(disease_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = []
    if os.path.isdir(pdf_dir):
        # If directory contains disease subdirectories, respect that structure
        if any(os.path.isdir(os.path.join(pdf_dir, d)) for d in DISEASE_TERMS):
            for disease in DISEASE_TERMS:
                disease_path = os.path.join(pdf_dir, disease)
                if os.path.isdir(disease_path) and (filter_disease is None or disease == filter_disease):
                    disease_pdfs = [os.path.join(disease_path, f) for f in os.listdir(disease_path) 
                                    if f.lower().endswith('.pdf')]
                    # Tag these files with their disease
                    pdf_files.extend([(pdf, disease) for pdf in disease_pdfs])
        else:
            # Just a flat directory of PDFs
            pdf_files = [(os.path.join(pdf_dir, f), None) for f in os.listdir(pdf_dir) 
                          if f.lower().endswith('.pdf')]
    else:
        logger.error(f"Not a valid directory: {pdf_dir}")
        return None
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return None
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    results = []
    summary = {
        "total_files": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "sentences_by_disease": {disease: 0 for disease in DISEASE_TERMS},
        "sentences_by_category": {cat: 0 for cat in ALL_CATEGORY_TERMS.keys()}
    }
        
    # Create corpus files
    corpus_file = os.path.join(output_dir, "wheat_disease_corpus.txt")
    with open(corpus_file, 'w', encoding='utf-8') as main_corpus:
        # Also create disease-specific corpus files
        disease_corpus_files = {}
        for disease in DISEASE_TERMS:
            path = os.path.join(output_dir, f"{disease}_corpus.txt")
            disease_corpus_files[disease] = open(path, 'w', encoding='utf-8')
        
        # Create category-specific corpus files
        category_corpus_files = {}
        for category in ALL_CATEGORY_TERMS:
            path = os.path.join(output_dir, f"{category}_corpus.txt")
            category_corpus_files[category] = open(path, 'w', encoding='utf-8')
        
        # Process PDFs
        for pdf_path, known_disease in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Determine output subdirectory
                result = process_pdf_file(pdf_path, output_dir)
                
                if not result:
                    summary["failed"] += 1
                    continue
                
                summary["successful"] += 1
                
                # Get disease focus or use the known disease from directory structure
                disease_focus = result["summary"]["disease_focus"]
                if not disease_focus and known_disease:
                    disease_focus = [known_disease]
                
                # # Write sentences to main corpus
                # for sentence in result["sentences"]:
                #     main_corpus.write(sentence + '\n')
                
                # Track counts
                all_categories = []
                if result["sentence_categories"]:
                    for cats in result["sentence_categories"]:
                        all_categories.extend(cats)
                    
                    # Count by category
                    category_counts = Counter(all_categories)
                    for category, count in category_counts.items():
                        summary["sentences_by_category"][category] = summary["sentences_by_category"].get(category, 0) + count
                
                # Write to disease-specific corpus files
                for disease in disease_focus:
                    # Count sentences by disease focus
                    summary["sentences_by_disease"][disease] = summary["sentences_by_disease"].get(disease, 0) + len(result["sentences"])
                    
                    # Write to disease corpus file
                    for sentence in result["sentences"]:
                        disease_corpus_files[disease].write(sentence + '\n')
                
                # Write to category-specific corpus files based on sentence categories
                if result["sentence_categories"]:
                    for i, sentence in enumerate(result["sentences"]):
                        if i < len(result["sentence_categories"]):
                            for category in result["sentence_categories"][i]:
                                if category in category_corpus_files:
                                    category_corpus_files[category].write(sentence + '\n')
                
                # Store result summary
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                summary["failed"] += 1
        
        
        # Deduplicate sentences in batches before writing to corpus files
        logger.info("Collecting all sentences for deduplication...")
        all_sentences = []
        for result in results:
            # Make sure the result contains sentences
            if isinstance(result, dict) and "sentences" in result:
                all_sentences.extend(result["sentences"])

        # Define batch size for memory-efficient processing
        BATCH_SIZE = 5000
        logger.info(f"Deduplicating {len(all_sentences)} sentences in batches of {BATCH_SIZE}...")
        deduplicated_sentences = []

        # Process in smaller batches
        for i in range(0, len(all_sentences), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(all_sentences))
            batch = all_sentences[i:batch_end]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(all_sentences)-1)//BATCH_SIZE + 1} with {len(batch)} sentences")
            batch_deduped = deduplicate_sentences(batch)
            deduplicated_sentences.extend(batch_deduped)

        # Second pass on the combined results (optional, with higher threshold)
        if len(deduplicated_sentences) > BATCH_SIZE:
            logger.info(f"Final deduplication pass on {len(deduplicated_sentences)} sentences")
            deduplicated_sentences = deduplicate_sentences(deduplicated_sentences, threshold=0.95)

        # Check if we have sentences before trying to write them
        if deduplicated_sentences:
            logger.info(f"Writing {len(deduplicated_sentences)} deduplicated sentences to corpus")
            for sentence in deduplicated_sentences:
                main_corpus.write(sentence + '\n')
        else:
            logger.warning("No sentences found for deduplication")

        # Close all corpus files
        for f in disease_corpus_files.values():
            f.close()
        for f in category_corpus_files.values():
            f.close()
    
    # Write processing summary
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Write detailed results
    results_file = os.path.join(output_dir, "processing_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Created corpus with {sum(summary['sentences_by_disease'].values())} sentences from {summary['successful']} PDFs")
    
    return summary

def train_word2vec_model(corpus_file, output_model_path=None, vector_size=200, window=5, min_count=3, epochs=15):
    """
    Train a Word2Vec model on the corpus
    
    Args:
        corpus_file: Path to corpus text file
        output_model_path: Where to save the model
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        epochs: Number of training iterations
    
    Returns:
        Trained Word2Vec model
    """
    try:
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess
    except ImportError:
        logger.error("Gensim is required for Word2Vec training. Install with 'pip install gensim'")
        return False
    
    logger.info("Preparing sentences for Word2Vec training...")
    sentences = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing sentences"):
            # Tokenize each sentence
            tokens = simple_preprocess(line)
            if tokens:
                sentences.append(tokens)
    
    logger.info(f"Training Word2Vec model on {len(sentences)} sentences...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=8,        # Parallel processing
        epochs=epochs,
        sg=1              # Skip-gram model (better for infrequent words)
    )
    
    if output_model_path:
        model.save(output_model_path)
        logger.info(f"Word2Vec model saved to {output_model_path}")
    
    # Print some example word similarities
    test_words = ['fungicide', 'resistance', 'wheat', 'rust', 'mildew', 'septoria', 
                 'management', 'cost', 'economic', 'yield', 'treatment', 'spray', 
                 'cultivar', 'application']
    
    logger.info("\nWord2Vec model test results:")
    found_words = 0
    
    
    for word in test_words:
        if word in model.wv:
            found_words += 1
            logger.info(f"\nWords similar to '{word}':")
            similar_words = model.wv.most_similar(word, topn=5)
            for similar_word, similarity in similar_words:
                logger.info(f"  {similar_word}: {similarity:.4f}")
        else:
            logger.info(f"\nWord '{word}' not in vocabulary")
    
    vocab_size = len(model.wv.index_to_key)
    logger.info(f"\nVocabulary size: {vocab_size} words")
    logger.info(f"Found {found_words}/{len(test_words)} test words in model vocabulary")
    
    return model

def analyze_corpus(corpus_dir):
    """
    Perform basic analysis of the generated corpus
    """
    corpus_files = {
        "main": os.path.join(corpus_dir, "wheat_disease_corpus.txt"),
        **{disease: os.path.join(corpus_dir, f"{disease}_corpus.txt") for disease in DISEASE_TERMS},
        **{category: os.path.join(corpus_dir, f"{category}_corpus.txt") for category in ALL_CATEGORY_TERMS}
    }
    
    stats = {}
    
    for name, path in corpus_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Count sentences
                num_sentences = len(lines)
                
                # Count words
                word_count = sum(len(line.split()) for line in lines)
                
                # Get a sample of sentences
                sample = lines[:5] if num_sentences >= 5 else lines
                
                stats[name] = {
                    "sentences": num_sentences,
                    "words": word_count,
                    "avg_words_per_sentence": round(word_count / num_sentences, 1) if num_sentences > 0 else 0,
                    "sample": sample
                }
    
    # Write stats to file
    stats_file = os.path.join(corpus_dir, "corpus_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("\nCorpus Statistics:")
    logger.info(f"Main corpus: {stats.get('main', {}).get('sentences', 0)} sentences, {stats.get('main', {}).get('words', 0)} words")
    
    logger.info("\nDisease-specific corpus statistics:")
    for disease in DISEASE_TERMS:
        if disease in stats:
            logger.info(f"- {disease}: {stats[disease]['sentences']} sentences, {stats[disease]['words']} words")
    
    logger.info("\nCategory-specific corpus statistics:")
    for category in ALL_CATEGORY_TERMS:
        if category in stats:
            logger.info(f"- {category}: {stats[category]['sentences']} sentences, {stats[category]['words']} words")
    
    return stats

def analyze_disease_tables(output_dir):
    """
    Analyze tables extracted from PDFs to identify useful patterns and data
    """
    logger.info("Analyzing extracted tables...")
    
    # Find all JSON files with extracted data
    json_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_data.json"):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        logger.warning("No data files found for table analysis")
        return None
    
    # Collect tables by disease
    tables_by_disease = {disease: [] for disease in DISEASE_TERMS}
    tables_by_disease["unknown"] = []
    
    # Track table statistics
    table_stats = {
        "total_tables": 0,
        "tables_by_disease": {disease: 0 for disease in DISEASE_TERMS},
        "tables_by_disease_unknown": 0,
        "tables_by_extraction_method": {"tabula": 0, "pymupdf": 0}
    }
    
    # Process each data file
    for json_path in tqdm(json_files, desc="Processing table data"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Skip files with no tables
                if not data.get("tables"):
                    continue
                
                # Get disease focus
                disease_focus = data.get("summary", {}).get("disease_focus", [])
                
                # Process tables
                for table in data["tables"]:
                    table_stats["total_tables"] += 1
                    table_stats["tables_by_extraction_method"][table["extraction_method"]] += 1
                    
                    # Add source file info to table
                    table["source_file"] = os.path.basename(json_path)
                    
                    # Assign table to disease
                    if disease_focus:
                        # If multiple diseases, assign to each one
                        for disease in disease_focus:
                            if disease in tables_by_disease:
                                tables_by_disease[disease].append(table)
                                table_stats["tables_by_disease"][disease] += 1
                    else:
                        tables_by_disease["unknown"].append(table)
                        table_stats["tables_by_disease_unknown"] += 1
                
        except Exception as e:
            logger.error(f"Error processing table data from {json_path}: {e}")
    
    # Write table statistics
    table_stats_file = os.path.join(output_dir, "table_stats.json")
    with open(table_stats_file, 'w', encoding='utf-8') as f:
        json.dump(table_stats, f, indent=2)
    
    # Write tables by disease to separate files
    for disease, tables in tables_by_disease.items():
        if tables:
            disease_tables_file = os.path.join(output_dir, f"{disease}_tables.json")
            with open(disease_tables_file, 'w', encoding='utf-8') as f:
                json.dump(tables, f, indent=2)
    
    logger.info(f"Table analysis complete: {table_stats['total_tables']} tables found")
    
    return table_stats
# Add this function to detect and remove duplicate sentences
def deduplicate_sentences(sentences, threshold=0.9):
    """
    Remove near-duplicate sentences from a list
    
    Args:
        sentences: List of sentences
        threshold: Similarity threshold (0-1) for considering sentences duplicates
        
    Returns:
        List of sentences with duplicates removed
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if len(sentences) <= 1:
        return sentences
    
    # Create TF-IDF vectors for all sentences
    vectorizer = TfidfVectorizer(min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except:
        # If vectorization fails, fall back to simple hash-based deduplication
        return list(dict.fromkeys(sentences))
    
    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Set diagonal to 0 to avoid self-matches
    np.fill_diagonal(similarity_matrix, 0)
    
    # Track which sentences to keep
    to_keep = [True] * len(sentences)
    
    # Identify duplicates
    for i in range(len(sentences)):
        if to_keep[i]:
            # Find all sentences that are too similar to this one
            for j in range(i+1, len(sentences)):
                if to_keep[j] and similarity_matrix[i, j] > threshold:
                    # Mark the shorter sentence as duplicate
                    if len(sentences[i]) >= len(sentences[j]):
                        to_keep[j] = False
                    else:
                        to_keep[i] = False
                        break
    
    # Keep only non-duplicate sentences
    deduplicated = [s for i, s in enumerate(sentences) if to_keep[i]]
    
    logger.info(f"Removed {len(sentences) - len(deduplicated)} duplicate sentences out of {len(sentences)}")
    return deduplicated

def main():
    parser = argparse.ArgumentParser(description="Enhanced Wheat Disease Text Extraction")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="wheat_disease_corpus", help="Output directory")
    parser.add_argument("--train", action="store_true", help="Train Word2Vec model after corpus creation")
    parser.add_argument("--disease", type=str, choices=list(DISEASE_TERMS.keys()), help="Process only a specific disease")
    parser.add_argument("--analyze", action="store_true", help="Analyze corpus after creation")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip PDF extraction and use existing corpus")
    parser.add_argument("--analyze_tables", action="store_true", help="Analyze extracted tables separately")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Define paths
    model_path = os.path.join(args.output_dir, "wheat_disease_word2vec.model")
    main_corpus_path = os.path.join(args.output_dir, "wheat_disease_corpus.txt")
    
    # Process PDFs and create corpus (unless skipped)
    if not args.skip_extraction:
        logger.info(f"Processing PDFs from {args.pdf_dir}")
        summary = process_pdf_directory(args.pdf_dir, args.output_dir, args.disease)
        
        if not summary:
            logger.error("PDF processing failed")
            return False
    else:
        logger.info("Skipping PDF extraction, using existing corpus files")
    
    # Analyze corpus if requested
    if args.analyze:
        if os.path.exists(main_corpus_path):
            logger.info("Analyzing corpus...")
            analyze_corpus(args.output_dir)
        else:
            logger.error("Cannot analyze corpus: main corpus file not found")
    
    # Analyze tables if requested
    if args.analyze_tables:
        logger.info("Analyzing tables...")
        analyze_disease_tables(args.output_dir)
    
    # Train Word2Vec model if requested
    if args.train:
        if os.path.exists(main_corpus_path):
            logger.info("Training Word2Vec model...")
            train_word2vec_model(main_corpus_path, model_path)
        else:
            logger.error("Cannot train model: main corpus file not found")
    
    logger.info("Processing complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)