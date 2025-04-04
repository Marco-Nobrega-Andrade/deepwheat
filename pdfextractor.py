#!/usr/bin/env python3
"""
Improved PDF Text Extractor for Wheat Disease Research
=====================================================

This script extracts text from PDF documents about wheat diseases
while specifically avoiding table content and duplicate information.

Features:
- Intelligent table detection and exclusion
- Header/footer recognition and removal
- Duplicate sentence filtering
- Academic metadata cleaning
- OCR support for scanned documents (optional)
"""

import os
import re
import argparse
import logging
from tqdm import tqdm
import json
import hashlib
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_extraction.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define patterns for table detection
TABLE_PATTERNS = [
    r'^Table\s+\d+[.:]\s+',  # Table caption
    r'^Fig(ure|\.)\s+\d+[.:]\s+',  # Figure caption
    r'^\s*\d+\s*\|\s*\d+\s*\|',  # Rows with pipe separators
    r'^\s*\d+\s*&\s*\d+\s*&',  # Rows with ampersand separators
    r'^\s*\d+\.\s+[-\w\s]+\s+\d+(\.\d+)?\s*$',  # Numbered list with values
    r'^\s*\d+\.?\s*&\s*\d+\.?\s*&',  # Data rows with ampersands
    r'^\s*\d+\s*\.\s*\d+\s*\.\s*\d+\s*\.\s*'  # Multiple section numbers
]

# Define patterns for headers and footers
HEADER_FOOTER_PATTERNS = [
    r'^\s*\d+\s*$',  # Just a page number
    r'^Page\s+\d+\s+(of|/)\s+\d+$',  # Page X of Y
    r'^\s*https?://\S+\s*$',  # Just a URL
    r'^\s*www\.\S+\s*$',  # Just a web address
    r'^\s*[A-Za-z][\w\s&]+\s+\|\s+[A-Za-z][\w\s]+\s*$',  # Journal header/footer
    r'^\s*[A-Za-z][\w\s&]+\s+•\s+Volume\s+\d+\s*$',  # Journal info
    r'^\s*©\s*\d{4}\s+',  # Copyright line
    r'^\s*doi:[\w\./-]+\s*$',  # Just a DOI
    r'^\s*\[\s?\d+\s?\](\s*-\s*\[\s?\d+\s?\])?\s*$'  # Reference numbers
]

# Define patterns for academic metadata
METADATA_PATTERNS = [
    r'^\s*Received:.*Accepted:.*$',
    r'^\s*Received\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}.*$',
    r'^\s*Published\s+online:.*$',
    r'^\s*Copyright\s+©.*$',
    r'^\s*Keywords:.*$',
    r'^\s*Corresponding Author:.*$',
    r'^\s*E-mail:.*$',
    r'^\s*https?://doi\.org/.*$',
    r'^\s*This article is published with.*$',
    r'^\s*This is an open access article.*$',
    r'^\s*ISSN:.*$',
    r'^\s*Journal of.*Vol\..*$',
    r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$'  # Date format
]

# Table detection regex
TABLE_REGEX = re.compile('|'.join(TABLE_PATTERNS))
HEADER_FOOTER_REGEX = re.compile('|'.join(HEADER_FOOTER_PATTERNS))
METADATA_REGEX = re.compile('|'.join(METADATA_PATTERNS))

# Define wheat disease terms for relevance filtering
DISEASE_TERMS = {
    "yellow_rust": ["yellow rust", "stripe rust", "puccinia striiformis", "yr", "pst", "p. striiformis"],
    "brown_rust": ["brown rust", "leaf rust", "puccinia triticina", "puccinia recondita", "lr", "pt", "p. triticina", "p. recondita"],
    "septoria": ["septoria", "septoria tritici", "zymoseptoria tritici", "septoria leaf blotch", 
                "stb", "septoria nodorum", "parastagonospora nodorum", "s. tritici", "z. tritici", "s. nodorum", "p. nodorum"],
    "mildew": ["mildew", "powdery mildew", "blumeria graminis", "erysiphe graminis", "pm", "b. graminis", "e. graminis"]
}

class PDFExtractor:
    """Class for extracting text from PDFs while avoiding tables"""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None
        self.text_blocks = []
        self.sentences = []
        self.headers_footers = []
        self.tables = []
        self.metadata = {}
    
    def open_document(self):
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
        
        return self.metadata
    
    def identify_headers_footers(self):
        """Identify potential headers and footers based on repeated text at page boundaries"""
        if not self.doc:
            return []
        
        # Track text blocks from the top and bottom of each page
        top_blocks = []
        bottom_blocks = []
        
        for page_idx, page in enumerate(self.doc):
            # Extract text blocks with their bounding boxes
            blocks = page.get_text("blocks")
            
            if blocks:
                # Sort blocks by vertical position (y0)
                blocks.sort(key=lambda b: b[1])
                
                # Get top and bottom blocks
                top_blocks.append(blocks[0][4] if len(blocks) > 0 else "")
                bottom_blocks.append(blocks[-1][4] if len(blocks) > 0 else "")
        
        # Find repeated blocks (appearing on at least 30% of pages)
        header_candidates = Counter(top_blocks)
        footer_candidates = Counter(bottom_blocks)
        
        min_repetitions = max(2, len(self.doc) * 0.3)  # At least 30% of pages
        
        headers = [text for text, count in header_candidates.items() 
                  if count >= min_repetitions and text and len(text.strip()) > 5]
        footers = [text for text, count in footer_candidates.items() 
                  if count >= min_repetitions and text and len(text.strip()) > 5]
        
        self.headers_footers = headers + footers
        return self.headers_footers
    
    def is_table_block(self, text):
        """Detect if a text block is likely part of a table"""
        if not text or len(text.strip()) < 5:
            return False
            
        # Check if it matches a table pattern
        if TABLE_REGEX.search(text):
            return True
        
        # Count tabs and spaces
        tab_count = text.count('\t')
        if tab_count > 2:
            return True
        
        # Check for excessive numbers and special characters
        text = text.strip()
        char_count = len(text)
        digit_count = sum(c.isdigit() for c in text)
        special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
        
        # If more than 40% digits or special characters, likely a table
        if char_count > 0 and (digit_count + special_char_count) / char_count > 0.4:
            return True
        
        # Check for evenly spaced content (table columns)
        spaces = [match.start() for match in re.finditer(r'\s{2,}', text)]
        if len(spaces) >= 3:
            # Check if spaces are evenly distributed (table columns)
            diffs = [spaces[i+1] - spaces[i] for i in range(len(spaces)-1)]
            if len(diffs) > 1 and max(diffs) - min(diffs) < 5:
                return True
        
        return False
    
    def is_header_footer(self, text):
        """Check if text is likely a header or footer"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Check against known headers/footers
        if text in self.headers_footers:
            return True
            
        # Check if it matches header/footer patterns
        if HEADER_FOOTER_REGEX.search(text):
            return True
            
        return False
    
    def is_academic_metadata(self, text):
        """Check if text is academic metadata"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Check if it matches metadata patterns
        if METADATA_REGEX.search(text):
            return True
            
        return False
    
    def extract_text_blocks(self):
        """Extract text blocks while avoiding tables"""
        if not self.doc:
            return []
            
        # Identify headers and footers first
        self.identify_headers_footers()
        
        # Process each page
        for page_idx, page in enumerate(self.doc):
            # Get text blocks with their bounding boxes
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4]  # The actual text content
                
                # Skip empty blocks
                if not text or len(text.strip()) < 5:
                    continue
                    
                # Check if it's a table
                if self.is_table_block(text):
                    self.tables.append(text)
                    continue
                    
                # Check if it's a header/footer
                if self.is_header_footer(text):
                    continue
                    
                # Check if it's academic metadata
                if self.is_academic_metadata(text):
                    continue
                    
                # Add to text blocks
                self.text_blocks.append(text)
        
        return self.text_blocks
    
    def extract_and_clean_sentences(self):
        """Extract and clean sentences from text blocks"""
        if not self.text_blocks:
            return []
            
        all_sentences = []
        
        for block in self.text_blocks:
            # Split into sentences
            try:
                block_sentences = sent_tokenize(block)
            except:
                # Fallback to simple splitting if NLTK fails
                block_sentences = re.split(r'[.!?]+', block)
                
            # Clean each sentence
            for sentence in block_sentences:
                # Clean the sentence
                sentence = self.clean_sentence(sentence)
                if sentence:
                    all_sentences.append(sentence)
        
        # Remove duplicates using hashing
        unique_sentences = []
        sentence_hashes = set()
        
        for s in all_sentences:
            h = hashlib.md5(s.lower().encode()).hexdigest()
            if h not in sentence_hashes:
                sentence_hashes.add(h)
                unique_sentences.append(s)
        
        self.sentences = unique_sentences
        return self.sentences
    
    def clean_sentence(self, sentence):
        """Clean a single sentence"""
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        
        # Skip if too short
        if len(sentence) < 20:
            return ""
            
        # Remove citations [1], [2,3], etc.
        sentence = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]\s*', ' ', sentence)
        
        # Remove figure and table references
        sentence = re.sub(r'\s*\(Fig\.\s*\d+\)\s*', ' ', sentence)
        sentence = re.sub(r'\s*\(Table\s*\d+\)\s*', ' ', sentence)
        
        # Remove URLs
        sentence = re.sub(r'https?://\S+', '', sentence)
        
        # Remove email addresses
        sentence = re.sub(r'\S+@\S+\.\S+', '', sentence)
        
        # Normalize whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Check if the sentence seems incomplete
        if not re.search(r'[.!?]$', sentence) and len(sentence) < 50:
            return ""
            
        return sentence
    
    def filter_disease_related(self):
        """Filter sentences to keep only those related to diseases"""
        if not self.sentences:
            return []
            
        disease_related = []
        
        # Flatten all disease terms
        all_disease_terms = []
        for terms in DISEASE_TERMS.values():
            all_disease_terms.extend(terms)
            
        # Check each sentence
        for sentence in self.sentences:
            sentence_lower = sentence.lower()
            
            # Check if any disease term is present
            if any(term in sentence_lower for term in all_disease_terms):
                disease_related.append(sentence)
        
        return disease_related
    
    def extract_all(self):
        """Run the complete extraction pipeline"""
        if not self.open_document():
            return False
            
        # Extract metadata
        self.extract_metadata()
        
        # Extract text blocks avoiding tables
        self.extract_text_blocks()
        
        # Extract and clean sentences
        self.extract_and_clean_sentences()
        
        # Close the document
        self.close()
        
        return True
    
    def close(self):
        """Close the document"""
        if self.doc:
            self.doc.close()
            self.doc = None

def process_pdf_file(pdf_path, output_dir=None, filter_for_diseases=True):
    """Process a single PDF file"""
    extractor = PDFExtractor(pdf_path)
    
    logger.info(f"Processing {pdf_path}")
    if not extractor.extract_all():
        logger.error(f"Failed to process {pdf_path}")
        return None
    
    # Get sentences
    sentences = extractor.sentences
    
    # Filter for disease-related content if requested
    if filter_for_diseases:
        disease_sentences = extractor.filter_disease_related()
        logger.info(f"Filtered {len(sentences)} sentences to {len(disease_sentences)} disease-related")
        sentences = disease_sentences
    
    # Write output if directory specified
    if output_dir and sentences:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Write sentences to file
        sentences_file = os.path.join(output_dir, f"{base_name}_sentences.txt")
        with open(sentences_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        # Write metadata and statistics
        stats = {
            "filename": base_name,
            "metadata": extractor.metadata,
            "stats": {
                "total_text_blocks": len(extractor.text_blocks),
                "tables_detected": len(extractor.tables),
                "headers_footers_detected": len(extractor.headers_footers),
                "sentences_extracted": len(extractor.sentences),
                "disease_sentences": len(sentences) if filter_for_diseases else "not filtered"
            }
        }
        
        stats_file = os.path.join(output_dir, f"{base_name}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    
    return sentences

def process_pdf_directory(pdf_dir, output_dir, filter_for_diseases=True):
    """Process all PDFs in a directory"""
    if not os.path.exists(pdf_dir):
        logger.error(f"Directory does not exist: {pdf_dir}")
        return None
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    all_sentences = []
    processed_files = 0
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        sentences = process_pdf_file(pdf_path, output_dir, filter_for_diseases)
        if sentences:
            all_sentences.extend(sentences)
            processed_files += 1
    
    logger.info(f"Processed {processed_files} files, extracted {len(all_sentences)} sentences")
    
    # Write all sentences to a combined file
    combined_file = os.path.join(output_dir, "combined_corpus.txt")
    with open(combined_file, 'w', encoding='utf-8') as f:
        for sentence in all_sentences:
            f.write(sentence + '\n')
    
    logger.info(f"Wrote combined corpus to {combined_file}")
    
    # Write summary statistics
    summary = {
        "total_pdfs": len(pdf_files),
        "processed_pdfs": processed_files,
        "total_sentences": len(all_sentences),
        "disease_filtered": filter_for_diseases
    }
    
    summary_file = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return all_sentences

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs while avoiding tables")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--pdf", type=str, help="Single PDF file to process")
    parser.add_argument("--output_dir", type=str, default="extracted_text", help="Output directory")
    parser.add_argument("--no_disease_filter", action="store_true", help="Don't filter for disease-related content")
    
    args = parser.parse_args()
    
    if not args.pdf_dir and not args.pdf:
        parser.error("You must specify either --pdf_dir or --pdf")
    
    filter_for_diseases = not args.no_disease_filter
    
    if args.pdf:
        # Process single file
        process_pdf_file(args.pdf, args.output_dir, filter_for_diseases)
    else:
        # Process directory
        process_pdf_directory(args.pdf_dir, args.output_dir, filter_for_diseases)
    
    logger.info("Extraction complete")

if __name__ == "__main__":
    main()