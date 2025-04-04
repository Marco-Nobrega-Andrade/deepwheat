#!/usr/bin/env python3
"""
UTF-8 File Validator
====================
Validates that a text file is properly encoded in UTF-8 and reports any issues.
"""

import argparse
import os
import sys

def validate_utf8_file(filepath):
    """
    Validates that a file is properly encoded in UTF-8.
    Reports line numbers with any encoding issues.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        bool: True if file is valid UTF-8, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return False
        
    problem_lines = []
    
    try:
        # First try to read the entire file at once
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                print(f"✓ File can be read as UTF-8")
                
                # Check for BOM
                if content.startswith('\ufeff'):
                    print("! File contains UTF-8 BOM (Byte Order Mark) at the beginning")
            except UnicodeDecodeError as e:
                print(f"✗ File failed complete UTF-8 validation: {e}")
        
        # Check line by line to find problem lines
        with open(filepath, 'rb') as f:
            for i, line_bytes in enumerate(f, 1):
                try:
                    # Try to decode the line as UTF-8
                    line_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    problem_lines.append(i)
    
    except Exception as e:
        print(f"Error checking file: {e}")
        return False
    
    if problem_lines:
        print(f"✗ Found {len(problem_lines)} lines with UTF-8 encoding issues:")
        for line_num in problem_lines[:10]:  # Show first 10 problem lines
            print(f"  - Line {line_num}")
        if len(problem_lines) > 10:
            print(f"  ... and {len(problem_lines) - 10} more")
            
        # Try to provide a hint about the actual encoding
        try:
            import chardet
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                print(f"\nDetected encoding: {result['encoding']} (confidence: {result['confidence']:.2f})")
                print("You may want to convert from this encoding to UTF-8")
        except ImportError:
            print("\nTip: Install 'chardet' package to detect the actual file encoding:")
            print("     pip install chardet")
            
        return False
    else:
        print("✓ All lines are valid UTF-8")
        
        # Check if there's a mixture of line endings
        with open(filepath, 'rb') as f:
            content = f.read()
            if b'\r\n' in content and b'\n' in content.replace(b'\r\n', b''):
                print("! File contains mixed line endings (both CRLF and LF)")
            elif b'\r\n' in content:
                print("ℹ File uses Windows (CRLF) line endings")
            else:
                print("ℹ File uses Unix (LF) line endings")
        
        return True

def convert_file_to_utf8(input_file, output_file=None, input_encoding=None):
    """
    Converts a file to UTF-8 encoding.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file (defaults to overwriting input file)
        input_encoding: Source encoding (if known)
        
    Returns:
        bool: True if conversion was successful
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".utf8"
    
    # Try to detect the encoding if not provided
    if input_encoding is None:
        try:
            import chardet
            with open(input_file, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                input_encoding = result['encoding']
                confidence = result['confidence']
                print(f"Detected encoding: {input_encoding} (confidence: {confidence:.2f})")
                
                if confidence < 0.7:
                    print("Warning: Low confidence in encoding detection!")
        except ImportError:
            print("Error: 'chardet' package not installed and no encoding specified.")
            print("Install with: pip install chardet")
            return False
    
    try:
        # Read from source encoding
        with open(input_file, 'r', encoding=input_encoding, errors='replace') as f_in:
            content = f_in.read()
        
        # Write with UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(content)
            
        print(f"✓ File converted to UTF-8 and saved to '{output_file}'")
        return True
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate and convert text files to UTF-8 encoding")
    parser.add_argument("file", help="File to validate or convert")
    parser.add_argument("--convert", action="store_true", help="Convert file to UTF-8")
    parser.add_argument("--from-encoding", help="Source encoding (if known)")
    parser.add_argument("--output", help="Output file for conversion (defaults to overwriting)")
    args = parser.parse_args()
    
    if args.convert:
        convert_file_to_utf8(args.file, args.output, args.from_encoding)
    else:
        validate_utf8_file(args.file)

if __name__ == "__main__":
    main()