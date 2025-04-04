#!/usr/bin/env python3
"""
PDF Text Extractor with Text Chunking and Enhanced Filtering
========================================================

- Processes text in smaller chunks to avoid memory issues
- Extracts clean English text from PDF documents with enhanced filtering
- Strictly enforces 12-30 word count limit
- Additional filters for references, table contents, and other irrelevant text
- No API calls to OpenAI (local processing only)
- Built-in timeout protection for problematic files
"""

import os
import re
import hashlib
import unicodedata
import argparse
import logging
import time
from pathlib import Path
import fitz  # PyMuPDF
import spacy
from langdetect import detect, DetectorFactory
from tqdm import tqdm

# Set seed for langdetect for reproducible results
DetectorFactory.seed = 0

# Load spaCy model with increased length limit
try:
    nlp = spacy.load("en_core_web_sm")
    # Don't adjust max_length here - we'll chunk the text instead
except OSError:
    import sys
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

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

# Enhanced patterns to detect unwanted content
UNWANTED_PATTERNS = [
    # Tables and figures
    r'^Table\s+\d+[.:]\s+',
    r'^Fig(ure|\.)\s+\d+[.:]\s+',
    r'^\s*\d+\s*\|\s*\d+\s*\|',
    r'^\s*\d+\s*&\s*\d+\s*&',
    r'\d+\s*[x×]\s*\d+',
    r'^\s*\d+\s*\.\s*\d+\s*$',
    
    # References and bibliographies
    r'et\s+al\.',
    r'\([12][0-9]{3}\)',  # Years in parentheses
    r'^\s*[A-Z][a-z]+\s+[A-Z]\.',  # Author name with initial
    r'[A-Z][a-z]+\s+[A-Z][a-z]*,\s+[A-Z]\.',  # Author name, initial format
    
    # Document structure elements
    r'^chapter\s+\d+',
    r'table\s+of\s+contents',
    r'appendix|annex',
    r'^abstract',
    r'^introduction',
    r'^conclusion',
    r'^references',
    r'^bibliography',
    r'^acknowledgements',
    
    # Supplementary materials
    r'see\s+attached',
    r'supplementary',
    r'additional\s+file',
    
    # Page references and section markers
    r'page\s+\d+',
    r'section\s+\d+',
    r'figure\s+\d+',
    
    # Special markers
    r'GS\d+',  # Growth stages
    r'vs\.',
    r'\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.',  # Ellipsis or dotted lines
    
    # Headers and footers
    r'^\s*\d+\s*$',  # Just a number (like page number)
]
UNWANTED_REGEX = re.compile('|'.join(UNWANTED_PATTERNS), re.IGNORECASE)

# Funding patterns
FUNDING_PATTERNS = [
    r'(support|fund)(ed)?\s+by\s+(grant|the|a)',
    r'(acknowledg|thank).{1,30}(support|fund|grant)',
    r'grant\s+(number|no\.)\s+[A-Z0-9\-]+',
    r'Ministry of Education',
    r'research\s+(was)?\s+supported',
    r'financial\s+support'
]
FUNDING_REGEX = re.compile('|'.join(FUNDING_PATTERNS), re.IGNORECASE)


class PDFTextExtractor:
    def __init__(self):
        self.sentence_hashes = set()
        self.MAX_CHUNK_SIZE = 100000  # Process text in 100K character chunks

    def extract_text_from_pdf(self, pdf_path, timeout=30):
        """Extract text from PDF with a simpler, more robust approach"""
        logger.info(f"Extracting text from {pdf_path}")
        
        # Check file size
        try:
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
            if file_size > 20:  # Skip files larger than 20MB
                logger.warning(f"Skipping large file ({file_size:.1f} MB): {pdf_path}")
                return {"valid_sentences": []}
        except:
            pass
            
        all_text = []
        
        try:
            # Use a simple timekeeper instead of signal-based timeout
            start_time = time.time()
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page with a time limit
            for page_num in range(len(doc)):
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout reached after processing {page_num} pages")
                    break
                    
                try:
                    # Get the page
                    page = doc[page_num]
                    
                    # Extract text as a single string (simpler, less prone to hanging)
                    text = page.get_text()
                    
                    # Only keep if we got something substantive
                    if text and len(text.strip()) > 20:
                        all_text.append(text)
                except Exception as e:
                    logger.warning(f"Error on page {page_num+1}: {str(e)}")
                    continue
            
            # Close the document
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            
        # Return the extracted text for sentence processing
        return self._process_text(all_text)

    def _process_text(self, text_chunks):
        """Process text in smaller chunks to avoid memory issues"""
        valid_sentences = []
        
        # Join all text chunks with a space
        text = " ".join(text_chunks)
        
        # Basic cleaning
        text = text.replace("\n", " ")
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        try:
            # Split into manageable chunks
            total_length = len(text)
            logger.info(f"Total text length: {total_length} characters")
            
            # Break into smaller chunks at sentence boundaries (periods)
            chunks = []
            current_pos = 0
            
            while current_pos < total_length:
                chunk_end = min(current_pos + self.MAX_CHUNK_SIZE, total_length)
                
                # Try to find a sentence boundary
                if chunk_end < total_length:
                    # Look for the last period within this chunk
                    last_period = text.rfind('. ', current_pos, chunk_end)
                    if last_period > current_pos:
                        chunk_end = last_period + 1  # Include the period
                
                chunks.append(text[current_pos:chunk_end])
                current_pos = chunk_end
            
            logger.info(f"Processing text in {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Use spaCy to split into sentences
                doc = nlp(chunk)
                
                # Extract sentences
                for sent in doc.sents:
                    # Get the sentence text
                    sentence = sent.text.strip()
                    
                    # Only process sentences within the word count range
                    word_count = len(sentence.split())
                    if word_count < 12 or word_count > 30:
                        continue
                    
                    # Clean and validate the sentence
                    cleaned = self._clean_sentence(sentence)
                    if cleaned:
                        valid_sentences.append(cleaned)
                        
        except Exception as e:
            logger.error(f"Error in sentence processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Found {len(valid_sentences)} valid sentences (12-30 words)")
        return {"valid_sentences": valid_sentences}

    def _clean_sentence(self, sentence):
        """Clean and validate a sentence with stricter filtering"""
        # Skip very short sentences or those outside our word count range
        word_count = len(sentence.split())
        if word_count < 12 or word_count > 30:
            return ""
        
        # Minimum character length
        if len(sentence) < 40:  # A good sentence usually has at least 40 chars
            return ""
            
        # Match against unwanted patterns
        if UNWANTED_REGEX.search(sentence):
            return ""
            
        # Skip funding acknowledgments
        if FUNDING_REGEX.search(sentence):
            return ""
            
        # Skip sentences with too many digits
        if sum(c.isdigit() for c in sentence) / len(sentence) > 0.1:
            return ""
            
        # Skip sentences with too many capital letters (likely headers or references)
        if sum(c.isupper() for c in sentence) / len(sentence) > 0.2:
            return ""
            
        # Skip sentences that don't start with capital letter or don't end with punctuation
        if not (sentence[0].isupper() and sentence[-1] in '.!?'):
            return ""
        
        # Skip sentences with too many symbols or special characters
        if sum(not c.isalnum() and not c.isspace() for c in sentence) / len(sentence) > 0.1:
            return ""
            
        # Skip references with multiple consecutive capitalized words (likely author names)
        if re.search(r'([A-Z][a-z]+\s+){2,}', sentence):
            return ""
            
        # Skip sentences containing "See attached" or similar
        if re.search(r'see (attached|table|appendix|supplementary)', sentence.lower()):
            return ""
            
        # Skip sentences with table of contents formatting
        if re.search(r'\.\.\.\.\.\.\.\.\.\.\.\d+', sentence):
            return ""
            
        # Skip non-English sentences
        try:
            if detect(sentence) != 'en':
                return ""
        except:
            pass
            
        # Check for duplicates
        h = hashlib.md5(sentence.lower().encode()).hexdigest()
        if h in self.sentence_hashes:
            return ""
        self.sentence_hashes.add(h)
        
        return sentence

    def process_pdf_directory(self, pdf_dir, output_dir):
        """Process a directory of PDFs and save the extracted corpus"""
        os.makedirs(output_dir, exist_ok=True)
        pdf_files = [os.path.join(dp, f) for dp, _, fn in os.walk(pdf_dir) for f in fn if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDFs")

        corpus_file = os.path.join(output_dir, "complete_corpus.txt")
        processed_file = os.path.join(output_dir, "processed_pdfs.txt")
        
        # Load existing progress
        processed_files = set()
        if os.path.exists(processed_file):
            with open(processed_file, "r", encoding="utf-8") as f:
                processed_files = set(line.strip() for line in f if line.strip())
        
        # Load existing sentences
        existing_sentences = set()
        if os.path.exists(corpus_file):
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_sentences.add(line)
            logger.info(f"Loaded {len(existing_sentences)} existing sentences")
        else:
            with open(corpus_file, "w", encoding="utf-8") as f:
                pass  # Create empty file
        
        total_sentences = len(existing_sentences)
        new_sentences = 0
        
        # Process each PDF
        for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs"), start=1):
            pdf_name = os.path.basename(pdf_path)
            
            # Skip already processed PDFs
            if pdf_name in processed_files:
                logger.info(f"[{i}/{len(pdf_files)}] Skipping already processed: {pdf_name}")
                continue
                
            logger.info(f"[{i}/{len(pdf_files)}] Processing {pdf_name}")
            
            try:
                # Extract text from PDF
                extracted = self.extract_text_from_pdf(pdf_path)
                valid_sentences = extracted["valid_sentences"]
                
                if valid_sentences:
                    # Filter out duplicates
                    unique_sentences = []
                    for sentence in valid_sentences:
                        # Double-check word count to be sure
                        word_count = len(sentence.split())
                        if word_count < 12 or word_count > 30:
                            continue
                            
                        if sentence not in existing_sentences:
                            unique_sentences.append(sentence)
                            existing_sentences.add(sentence)
                    
                    # Save unique sentences
                    if unique_sentences:
                        with open(corpus_file, "a", encoding="utf-8") as f:
                            for sentence in unique_sentences:
                                f.write(sentence + "\n")
                                
                        new_sentences += len(unique_sentences)
                        logger.info(f"Added {len(unique_sentences)} new sentences from {pdf_name}")
                    else:
                        logger.info(f"No new sentences found in {pdf_name}")
                else:
                    logger.info(f"No valid sentences found in {pdf_name}")
                
                # Mark as processed
                with open(processed_file, "a", encoding="utf-8") as f:
                    f.write(pdf_name + "\n")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # Still mark as processed
                with open(processed_file, "a", encoding="utf-8") as f:
                    f.write(pdf_name + "\n")
        
        logger.info(f"Corpus complete. Total sentences: {total_sentences + new_sentences} ({new_sentences} new)")
        return corpus_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="extracted_corpus", help="Output directory for corpus")
    args = parser.parse_args()

    extractor = PDFTextExtractor()
    extractor.process_pdf_directory(args.pdf_dir, args.output_dir)

if __name__ == "__main__":
    main()