#!/usr/bin/env python3
"""
Corpus Normalization Script
===========================

Normalizes text corpus files by:
1. Standardizing Unicode characters
2. Replacing problematic characters with ASCII equivalents
3. Removing control characters
4. Fixing common encoding issues

This helps create a clean corpus for NLP tasks like Word2Vec training.
"""

import os
import re
import argparse
import unicodedata
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("normalize_corpus.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_text(text):
    """
    Apply comprehensive text normalization
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text
    """
    if not text:
        return text
        
    # 1. Normalize unicode (NFC - canonical composition)
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # 3. Replace problematic characters based on the debugging report
    replacements = {
        # Quotes
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u201f': '"',  # Double high-reversed-9 quotation mark
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201b': "'",  # Single high-reversed-9 quotation mark
        '\u201e': '"',  # Double low-9 quotation mark
        '\u00ab': '"',  # Left-pointing double angle quotation mark
        '\u00bb': '"',  # Right-pointing double angle quotation mark
        
        # Dashes and hyphens
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2010': '-',  # Hyphen
        '\u2011': '-',  # Non-breaking hyphen
        '\u2012': '-',  # Figure dash
        '\u00ad': '',   # Soft hyphen (remove it)
        
        # Spaces
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero width space (remove it)
        
        # Math symbols
        '\u00b1': '+/-',  # Plus-minus sign
        '\u00d7': 'x',    # Multiplication sign
        '\u2212': '-',    # Minus sign
        '\u2264': '<=',   # Less-than or equal to
        '\u2265': '>=',   # Greater-than or equal to
        '\u2248': '~=',   # Almost equal to
        '\u2260': '!=',   # Not equal to
        '\u226b': '>>',   # Much greater-than
        '\u2261': '==',   # Identical to
        '\u2217': '*',    # Asterisk operator
        '\u221a': 'sqrt', # Square root
        '\u223c': '~',    # Tilde operator
        '\u2211': 'sum',  # N-ary summation
        '\u2206': 'delta',# Increment
        
        # Special symbols
        '\u00b0': ' degrees', # Degree sign
        '\u2103': ' C',      # Degree Celsius
        '\u2022': '*',       # Bullet
        '\u25e6': 'o',       # White bullet
        '\u2666': 'diamond', # Black diamond suit
        '\u25cf': '*',       # Black circle
        '\u26ac': 'o',       # Medium small white circle
        '\u00ae': '(R)',     # Registered sign
        '\u2122': '(TM)',    # Trade mark sign
        '\u00a9': '(C)',     # Copyright sign
        '\u00a3': 'GBP',     # Pound sign
        '\u20ac': 'EUR',     # Euro sign
        '\u00a5': 'JPY',     # Yen sign
        '\u2026': '...',     # Horizontal ellipsis
        '\u2192': '->',      # Rightwards arrow
        '\u2713': 'check',   # Check mark
        '\u2020': '+',       # Dagger
        '\u00b6': 'P',       # Pilcrow sign
        '\u00a1': '!',       # Inverted exclamation mark
        '\u00bf': '?',       # Inverted question mark
        '\u00a6': '|',       # Broken bar
        '\u203e': '-',       # Overline
        
        # Modifier and combining characters
        '\u0384': '',        # Greek tonos (remove)
        '\u02da': '',        # Ring above (remove)
        '\u0315': '',        # Combining comma above right (remove)
        '\u02dc': '~',       # Small tilde
        '\u02c2': '<',       # Modifier letter left arrowhead
        '\u02c3': '>',       # Modifier letter right arrowhead
        '\u02d8': '',        # Breve (remove)
        '\u02db': '',        # Ogonek (remove)
        '\u00b8': '',        # Cedilla (remove)
        '\u20dd': '',        # Combining enclosing circle (remove)
        '\u0338': '',        # Combining long solidus overlay (remove)
        '\u02d9': '',        # Dot above (remove)
        '\u0336': '',        # Combining long stroke overlay (remove)
        '\u00b4': '',        # Acute accent (remove)
        '\u00af': '',        # Macron (remove)
        '\u0060': '',        # Grave accent (remove)
        
        # Mathematical symbols
        '\u2208': 'in',      # Element of
        '\u25b3': 'triangle', # White up-pointing triangle
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Handle special characters that were identified as "UNKNOWN" in debug
    # Replace any remaining non-ASCII characters with space or remove them
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()

def normalize_corpus_file(input_file, output_file, preserve_empty_lines=False):
    """
    Normalize all text in a corpus file
    
    Args:
        input_file: Path to input corpus file
        output_file: Path to output normalized file
        preserve_empty_lines: Whether to keep empty lines in output
        
    Returns:
        Tuple of (total_lines, modified_lines)
    """
    logger.info(f"Normalizing corpus file: {input_file} -> {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return 0, 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_lines = 0
    modified_lines = 0
    
    try:
        # Get total line count for progress bar
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, total=total_lines, desc="Normalizing text"):
                total_lines += 1
                original_line = line.rstrip('\n')
                normalized_line = normalize_text(original_line)
                
                # Check if line was modified
                if normalized_line != original_line:
                    modified_lines += 1
                
                # Write normalized line
                if normalized_line or preserve_empty_lines:
                    f_out.write(normalized_line + '\n')
                    
        logger.info(f"Normalized {total_lines} lines, modified {modified_lines} lines ({modified_lines/total_lines*100:.2f}%)")
        return total_lines, modified_lines
        
    except Exception as e:
        logger.error(f"Error normalizing corpus file: {e}")
        return 0, 0

def normalize_corpus_directory(input_dir, output_dir, file_extension='.txt', recursive=False, preserve_empty_lines=False):
    """
    Normalize all corpus files in a directory
    
    Args:
        input_dir: Path to directory containing corpus files
        output_dir: Path to output directory for normalized files
        file_extension: File extension to filter corpus files
        recursive: Whether to process subdirectories
        preserve_empty_lines: Whether to keep empty lines in output
        
    Returns:
        Tuple of (processed_files, total_modified_lines)
    """
    logger.info(f"Normalizing corpus directory: {input_dir} -> {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0, 0
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_files = 0
    total_modified_lines = 0
    
    # Process files
    for root, dirs, files in os.walk(input_dir):
        # Skip if not recursive and not in root directory
        if not recursive and root != input_dir:
            continue
            
        # Get relative path to create same structure in output directory
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == '.':
            rel_path = ''
            
        # Create corresponding output directory
        curr_output_dir = os.path.join(output_dir, rel_path)
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)
            
        # Process each file
        for file in files:
            if file.endswith(file_extension):
                input_file = os.path.join(root, file)
                output_file = os.path.join(curr_output_dir, file)
                
                _, modified_lines = normalize_corpus_file(
                    input_file, output_file, preserve_empty_lines)
                
                processed_files += 1
                total_modified_lines += modified_lines
    
    logger.info(f"Processed {processed_files} files, modified {total_modified_lines} lines total")
    return processed_files, total_modified_lines

def analyze_text_sample(text):
    """Analyze a sample of text to identify potential normalization issues"""
    if not text:
        return
        
    char_counts = {}
    for char in text:
        if ord(char) > 127:  # Non-ASCII
            if char not in char_counts:
                char_counts[char] = 0
            char_counts[char] += 1
    
    if char_counts:
        logger.info("Non-ASCII characters found in sample:")
        for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = "UNKNOWN"
                
            logger.info(f"  '{char}' (U+{ord(char):04X}, {char_name}): {count} occurrences")
    
    # Show before/after sample
    norm_text = normalize_text(text)
    if text != norm_text:
        logger.info("\nSample text before normalization:")
        logger.info(text[:500] + ("..." if len(text) > 500 else ""))
        logger.info("\nSample text after normalization:")
        logger.info(norm_text[:500] + ("..." if len(norm_text) > 500 else ""))

def main():
    parser = argparse.ArgumentParser(description="Normalize text corpus files for NLP processing")
    parser.add_argument("--input", required=True, help="Input corpus file or directory")
    parser.add_argument("--output", required=True, help="Output normalized file or directory")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--preserve-empty", action="store_true", help="Preserve empty lines in output")
    parser.add_argument("--ext", default=".txt", help="File extension to process (default: .txt)")
    parser.add_argument("--analyze", action="store_true", help="Analyze a sample before processing")
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        return
    
    # Analyze sample if requested
    if args.analyze and os.path.isfile(args.input):
        logger.info(f"Analyzing sample from: {args.input}")
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                sample = f.read(10000)  # Read first 10KB
                analyze_text_sample(sample)
        except Exception as e:
            logger.error(f"Error analyzing sample: {e}")
    
    # Process file or directory
    if os.path.isfile(args.input):
        normalize_corpus_file(args.input, args.output, args.preserve_empty)
    else:
        normalize_corpus_directory(
            args.input, args.output, args.ext, args.recursive, args.preserve_empty)
    
    logger.info("Corpus normalization complete!")

if __name__ == "__main__":
    main()