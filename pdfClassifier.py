import os
import re
import json
import csv
import time
import fitz
import spacy
import tabula
from tqdm import tqdm

# Target diseases to search for (using various naming conventions)
DISEASES = {
    "yellow_rust": ["yellow rust", "stripe rust", "puccinia striiformis", "yr", "pst"],
    "brown_rust": ["brown rust", "leaf rust", "puccinia triticina", "puccinia recondita", "lr", "pt"],
    "septoria": ["septoria", "septoria tritici", "zymoseptoria tritici", "septoria leaf blotch", "stb", "septoria nodorum", "parastagonospora nodorum"],
    "mildew": ["mildew", "powdery mildew", "blumeria graminis", "erysiphe graminis", "pm"]
}

# Keywords for each information category
INFO_CATEGORIES = {
    "disease_detection": ["detected", "identified", "diagnosed", "outbreak", "infection", "symptoms", "occurrence", 
                         "prevalent", "incidence", "severity", "spread", "epidemic", "identified", "confirmed", 
                         "surveillance", "monitoring", "survey"],
    "treatment": ["treatment", "fungicide", "control measure", "management", "spray", "prevention", "recommendation", 
                 "application", "timing", "rate", "efficacy", "resistance", "active ingredient", "chemical control", 
                 "biological control", "integrated management", "cultural practice", "rotation", "varieties", 
                 "resistant cultivar", "tolerant", "seed treatment"],
    "crop_loss": ["yield loss", "crop loss", "damage", "reduction", "decrease", "impact on yield", "production loss", 
                 "harvest loss", "percentage loss", "tonnage", "bushels", "yield potential", "yield penalty", 
                 "crop damage", "production impact", "harvest reduction", "grain quality", "test weight", 
                 "thousand grain weight", "hectolitre weight"],
    "financial_impact": ["economic", "financial", "cost", "loss", "impact", "million", "billion", "price", "market", 
                        "income", "revenue", "profit", "margin", "expense", "investment", "return", "budget", 
                        "insurance", "compensation", "subsidy", "gdp", "economy", "trade", "export", "import", 
                        "dollar", "euro", "pound", "ruble", "yuan", "currency"]
}

# Validation patterns for each category
VALIDATION_PATTERNS = {
    "disease_detection": r'(found|observed|detected|reported|symptoms|signs|present|infected|identified|occurred|confirmed|diagnosed|developed)',
    "treatment": r'(apply|spray|use|control|manage|effective|resistant|fungicide|treatment|protect|prevent|application|timing|program)',
    "crop_loss": r'(loss|yield|damage|reduced|decreased|impact|affected|decline|reduction|diminish|lower|compromise)',
    "financial_impact": r'(cost|economic|financial|money|million|dollars|euros|pounds|price|value|revenue|profit|margin)'
}

# Directory paths
PDF_DIR = "wheat_disease_pdfs"
OUTPUT_DIR = "wheat_disease_pdf_data"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv_output")

# Initialize spaCy for better sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
    USING_SPACY = True
    print("Using spaCy for improved sentence detection")
except:
    USING_SPACY = False
    print("spaCy not available, falling back to regex-based sentence detection")

def create_output_dirs():
    """Create output directories if they don't exist"""
    for directory in [OUTPUT_DIR, CSV_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def split_into_sentences(text):
    """Split text into sentences using spaCy if available, otherwise regex"""
    if USING_SPACY:
        # Use spaCy for better sentence detection
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        # Fallback: basic regex-based sentence splitting
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def identify_disease_in_text(text):
    """Identify which disease(s) are mentioned in the text"""
    text_lower = text.lower()
    detected_diseases = []
    
    for disease_key, disease_terms in DISEASES.items():
        if any(term in text_lower for term in disease_terms):
            detected_diseases.append(disease_key)
    
    return detected_diseases

def extract_information_from_pdf(pdf_path, include_tables=True, disease_terms_only=True):
    """Extract information from a PDF and categorize by disease and information type"""
    # Extract text with table handling
    text = extract_text_from_pdf_with_tables(pdf_path, include_tables, disease_terms_only)
    if not text:
        return None
    
    # Identify diseases
    detected_diseases = identify_disease_in_text(text)
    if not detected_diseases:
        return None
    
    # Initialize results structure
    file_name = os.path.basename(pdf_path)
    result = {
        "file_name": file_name,
        "path": pdf_path,
        "diseases_info": []
    }
    
    # Split into sentences
    all_sentences = split_into_sentences(text)
    
    # Process each detected disease
    for disease in detected_diseases:
        disease_terms = DISEASES[disease]
        
        disease_info = {
            "disease": disease,
            "detection": [],
            "treatment": [],
            "crop_loss": [],
            "financial_impact": []
        }
        
        # Find sentences related to this disease
        for sentence in all_sentences:
            sentence_lower = sentence.lower()
            
            # Skip very short sentences or bibliographic references
            if len(sentence) < 20 or sentence.startswith("et al.") or re.match(r'^\d+\.\s', sentence):
                continue
                
            # Check if sentence mentions the disease
            if any(term in sentence_lower for term in disease_terms):
                # Check each information category
                for category, keywords in INFO_CATEGORIES.items():
                    if any(keyword in sentence_lower for keyword in keywords):
                        category_key = category.replace("disease_", "")
                        
                        # Additional validation using more specific patterns
                        validation_pattern = VALIDATION_PATTERNS[category]
                        if re.search(validation_pattern, sentence_lower):
                            # If validated, add to the appropriate category
                            if sentence not in disease_info[category_key]:
                                disease_info[category_key].append(sentence)
        
        # Check if we found any useful information
        has_info = False
        for key, values in disease_info.items():
            if key != "disease" and values:
                has_info = True
                break
        
        if has_info:
            result["diseases_info"].append(disease_info)
    
    if result["diseases_info"]:
        return result
    
    return None

def process_pdfs(pdf_directory, include_tables=True, disease_terms_only=True):
    """Process all PDFs in a directory and categorize their content"""
    # List all PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    all_results = []
    
    # Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        result = extract_information_from_pdf(pdf_path, include_tables, disease_terms_only)
        if result:
            all_results.append(result)
    
    return all_results

def export_to_csv(results):
    """Export the categorized data to CSV files"""
    # Create a CSV per disease
    for disease_key in DISEASES.keys():
        disease_name = disease_key.replace("_", " ").title()
        
        # Create main dataset CSV with all text and labels
        csv_file = os.path.join(CSV_DIR, f"{disease_key}_dataset.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'detection', 'treatment', 'crop_loss', 'financial_impact'])
            
            for result in results:
                for disease_info in result.get("diseases_info", []):
                    if disease_info["disease"] == disease_key:
                        # For each information category
                        for detection in disease_info.get("detection", []):
                            writer.writerow([detection, 1, 0, 0, 0])
                        for treatment in disease_info.get("treatment", []):
                            writer.writerow([treatment, 0, 1, 0, 0])
                        for crop_loss in disease_info.get("crop_loss", []):
                            writer.writerow([crop_loss, 0, 0, 1, 0])
                        for financial in disease_info.get("financial_impact", []):
                            writer.writerow([financial, 0, 0, 0, 1])
        
        print(f"Exported dataset for {disease_name} to {csv_file}")
    
    # Create a single CSV with all structured data
    structured_csv = os.path.join(CSV_DIR, "all_wheat_diseases_structured.csv")
    with open(structured_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['disease', 'category', 'text', 'source_file'])
        
        for result in results:
            source_file = result.get("file_name", "")
            
            for disease_info in result.get("diseases_info", []):
                disease = disease_info["disease"].replace("_", " ").title()
                
                for category, texts in disease_info.items():
                    if category != "disease" and texts:
                        for text in texts:
                            writer.writerow([
                                disease,
                                category,
                                text,
                                source_file
                            ])
    
    print(f"Exported structured data to {structured_csv}")

def export_for_word2vec(results):
    """Export data in formats suitable for Word2Vec training"""
    # Create corpus files for each disease and category
    for disease_key in DISEASES.keys():
        disease_name = disease_key.replace("_", " ")
        
        # Create a directory for this disease
        disease_dir = os.path.join(OUTPUT_DIR, f"{disease_key}_corpus")
        if not os.path.exists(disease_dir):
            os.makedirs(disease_dir)
        
        # Create corpus files for each category
        for category in ["detection", "treatment", "crop_loss", "financial_impact"]:
            corpus_file = os.path.join(disease_dir, f"{category}_corpus.txt")
            
            with open(corpus_file, 'w', encoding='utf-8') as f:
                for result in results:
                    for disease_info in result.get("diseases_info", []):
                        if disease_info["disease"] == disease_key:
                            for text in disease_info.get(category, []):
                                f.write(text + "\n")
    
    # Create a combined corpus file with category labels
    labeled_corpus = os.path.join(OUTPUT_DIR, "labeled_disease_corpus.txt")
    with open(labeled_corpus, 'w', encoding='utf-8') as f:
        for result in results:
            for disease_info in result.get("diseases_info", []):
                disease = disease_info["disease"]
                
                for category, texts in disease_info.items():
                    if category != "disease" and texts:
                        for text in texts:
                            # Format: __LABEL__category __LABEL__disease text
                            f.write(f"__LABEL__{category} __LABEL__{disease} {text}\n")
    
    # Create a simple text corpus with all sentences
    all_sentences = os.path.join(OUTPUT_DIR, "all_disease_sentences.txt")
    with open(all_sentences, 'w', encoding='utf-8') as f:
        for result in results:
            for disease_info in result.get("diseases_info", []):
                for category, texts in disease_info.items():
                    if category != "disease" and texts:
                        for text in texts:
                            f.write(text + "\n")
    
    print(f"Exported Word2Vec training files to {OUTPUT_DIR}")
    print(f"Created labeled corpus: {labeled_corpus}")
    print(f"Created full sentence corpus: {all_sentences}")

def analyze_data_quality(results):
    """Analyze the quality and distribution of the extracted data"""
    # Count reports per disease
    disease_counts = {disease: 0 for disease in DISEASES.keys()}
    category_counts = {
        "detection": 0,
        "treatment": 0,
        "crop_loss": 0,
        "financial_impact": 0
    }
    
    total_sentences = 0
    file_counts = {}
    
    for result in results:
        file_name = result.get("file_name", "Unknown")
        file_counts[file_name] = {
            "total_sentences": 0,
            "diseases": []
        }
        
        for disease_info in result.get("diseases_info", []):
            disease = disease_info["disease"]
            disease_counts[disease] += 1
            file_counts[file_name]["diseases"].append(disease)
            
            file_sentence_count = 0
            for category in category_counts.keys():
                sentences = disease_info.get(category, [])
                category_counts[category] += len(sentences)
                file_sentence_count += len(sentences)
                total_sentences += len(sentences)
            
            file_counts[file_name]["total_sentences"] += file_sentence_count
    
    # Print analysis results
    print("\n=== DATA QUALITY ANALYSIS ===")
    print(f"Total PDFs processed: {len(results)}")
    print(f"Total sentences extracted: {total_sentences}")
    
    print("\nDisease distribution:")
    for disease, count in disease_counts.items():
        disease_name = disease.replace("_", " ").title()
        print(f"- {disease_name}: {count} PDFs")
    
    print("\nInformation category distribution:")
    for category, count in category_counts.items():
        print(f"- {category}: {count} sentences")
    
    print("\nTop 5 most information-rich PDFs:")
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1]["total_sentences"], reverse=True)
    for file_name, stats in sorted_files[:5]:
        disease_names = [d.replace("_", " ").title() for d in stats["diseases"]]
        print(f"- {file_name}: {stats['total_sentences']} sentences, Diseases: {', '.join(disease_names)}")
    
    # Return whether we have sufficient data for Word2Vec training
    has_sufficient_data = total_sentences >= 200  # Minimum threshold for effective embeddings
    
    print("\nWord2Vec Training Readiness:")
    if has_sufficient_data:
        print("✓ Sufficient data collected for Word2Vec training")
    else:
        print(f"✗ Insufficient data for optimal Word2Vec training (have {total_sentences}, recommend at least 200)")
    
    return has_sufficient_data, disease_counts, category_counts

def extract_text_from_pdf_with_tables(pdf_path, include_tables=True, disease_terms_only=True):
    """Extract text from PDF with better table handling"""
    try:
        # Basic text extraction (keeps existing code)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Process tables separately if requested
        tables_text = ""
        if include_tables:
            try:
                # Extract tables using tabula
                tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
                
                # Process each table
                for i, table in enumerate(tables):
                    # Convert table to string representation
                    table_str = table.to_string(index=False)
                    
                    # Only include tables with disease terms if that filter is on
                    if not disease_terms_only or any(disease_term in table_str.lower() 
                                                    for disease_list in DISEASES.values() 
                                                    for disease_term in disease_list):
                        tables_text += f"\nTable {i+1}:\n{table_str}\n"
            except Exception as e:
                print(f"Warning: Table extraction failed for {pdf_path}: {e}")
        
        # Combine regular text with table text
        full_text = text + tables_text
        
        # Clean up the text
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    

def main(include_tables=True, disease_terms_only=True):
    """Main function to process PDFs and categorize their content"""
    print("===============================================")
    print("   WHEAT DISEASE PDF CATEGORIZER")
    print("===============================================")
    print("Configuration:")
    print(f"- Include tables: {'Yes' if include_tables else 'No'}")
    print(f"- Filter tables for disease terms: {'Yes' if disease_terms_only else 'No'}")

    for disease_key in DISEASES.keys():
        disease_name = disease_key.replace("_", " ").title()
        print(f"- {disease_name}")
    print("-----------------------------------------------")
    
    # Start timing
    start_time = time.time()
    
    # Create output directories
    create_output_dirs()
    
    # Process PDFs
    results = process_pdfs(PDF_DIR, include_tables, disease_terms_only)
    
    if not results:
        print("No results extracted from PDFs. Check that your PDFs contain relevant disease information.")
        return
    
    # Save raw collected data
    raw_data_file = os.path.join(OUTPUT_DIR, "wheat_disease_pdf_data.json")
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved data from {len(results)} PDFs to {raw_data_file}")
    
    # Analyze data quality and distribution
    has_sufficient_data, disease_counts, category_counts = analyze_data_quality(results)
    
    # Export data to CSV
    export_to_csv(results)
    
    # Export data for Word2Vec training
    export_for_word2vec(results)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print("\n-----------------------------------------------")
    print(f"PDF processing completed in {minutes}m {seconds}s")
    print(f"Processed {len(results)} PDFs with information about wheat diseases")
    
    # Summary of findings for each disease
    print("\nSummary of findings:")
    for disease_key, count in disease_counts.items():
        disease_name = disease_key.replace("_", " ").title()
        category_sentence_counts = {}
        for result in results:
            for disease_info in result.get("diseases_info", []):
                if disease_info["disease"] == disease_key:
                    for category in ["detection", "treatment", "crop_loss", "financial_impact"]:
                        current = category_sentence_counts.get(category, 0)
                        category_sentence_counts[category] = current + len(disease_info.get(category, []))
        
        print(f"- {disease_name}: {count} PDFs")
        for category, sent_count in category_sentence_counts.items():
            print(f"  * {category}: {sent_count} sentences")
    
    # Provide next steps based on data quality
    print("\nNext steps:")
    if has_sufficient_data:
        print("1. Use the generated corpus files for Word2Vec training")
        print("2. Try fastText with the labeled_disease_corpus.txt for classification")
        print("3. Consider manual validation of a sample of the extracted sentences")
    else:
        print("1. Add more PDF documents to improve data coverage")
        print("2. Consider supplementing with web-scraped data")
        print("3. Focus on adding documents about diseases with fewer extractions")

if __name__ == "__main__":
    main(include_tables=False, disease_terms_only=True)