import ollama
import time
import re

# Settings
INPUT_FILE = '../extracted_corpus/combined_corpus.txt'
OUTPUT_FILE = '../extracted_corpus/ollama_filtered_corpus.txt'
BATCH_SIZE = 10  # Reduced for better handling
OLLAMA_MODEL = 'llama3'

# Define the enhanced prompt with explicit instructions about output format
PROMPT_TEMPLATE = """
You are processing a batch of sentences extracted from agricultural research papers.

Your task:
- Keep only the sentences that are relevant to:
    - wheat diseases
    - wheat disease treatments
    - wheat yield loss
    - economic impact of wheat diseases
    - wheat disease management
- Sentences must have more than 4 words.
- Ignore sentences related to citations, acknowledgments, website URLs, or index/table of contents.
- If a sentence mixes relevant and irrelevant parts, extract and keep **only the relevant part**.

You must EXCLUDE:
- Sentences about genetics, genes, molecular markers, or genomics
- Sentences about DNA or RNA sequences or analysis
- Sentences about gene expression or gene mapping
- Sentences about QTLs (Quantitative Trait Loci)
- Sentences about genome sequencing or genetic engineering
- Non-English sentences
- Sentences describing research procedures, experimental designs, or methodologies
- Sentences listing technical parameters (e.g., R² values, RMSE values, confidence intervals)
- Paper structure mentions like "Methods", "Results", "Discussion", etc.
- Exclude sentences discussing how wheat diseases affect countries, regions, continents, or geopolitical areas (e.g., "wheat production risk in China", "31 countries affected", "impact in South America").
- Exclude sentences discussing meteorological conditions, climate factors, weather patterns, temperature, or rainfall related to disease spread.
- Exclude sentences about experimental design, remote sensing, spectral reflectance, satellite data, hyperspectral imagery, or machine learning models.
- Exclude any sentence related to genetic analysis, molecular markers, gene expression, QTLs, DNA/RNA sequencing, or genomics.
- Exclude any technical parameters like R² values, RMSE values, or error rates.
- Exclude any reference to papers, sections ("Methods", "Results", "Discussion", etc.), or citations.
- Exclude non-English sentences.
- Exclude general statistics not directly tied to a specific wheat disease impact (e.g., "world food production losses").
- Exclude generic statements like "Here are the relevant sentences:", "Summary:", "Results show:", etc.

Important context:
- Only sentences describing wheat diseases, their symptoms, their treatment, yield losses, economic impact, or disease management practices are allowed.
- Examples of wheat diseases: septoria, yellow rust, brown rust, stripe rust, leaf rust, glume blotch, powdery mildew, mildew, stagonospora, tan spot, ergot, fusarium head blight.
    - Sentences mentioning any of these diseases are relevant even if "wheat" is not explicitly mentioned.
- Treatments and management methods of interest include:
    - Fungicide application (e.g., triazoles, strobilurins, SDHIs)
    - Use of resistant wheat varieties
    - Crop rotation practices
    - Seed treatment with fungicides
    - Biological control methods
    - Irrigation and drainage management
    - Timing of sowing to avoid peak disease periods
    - Field residue management
    - Integrated Disease Management (IDM) strategies
- Impact sentences of interest:
    - Grain yield reduction (lower wheat production)
    - Grain quality loss
    - Financial or economic losses due to wheat diseases
    - Cost or expense related to disease management.

Output MUST STRICTLY FOLLOW these requirements:
1. ONLY output the relevant sentences, each on its own line.
2. DO NOT include any explanations, titles, headers, labels (e.g., "Here are the relevant sentences:"), bullet points, numbering, or any other formatting.
3. DO NOT generate any commentary, summaries, meta-text, or instructions.
4. If a sentence contains both relevant and irrelevant information, KEEP ONLY the relevant part (crop non-relevant parts cleanly).
5. If no relevant part remains after cleaning, discard the entire sentence.
6. Do not create new sentences; only rephrase minimally to clean irrelevant parts if needed.
7. Output must be plain text, each sentence separated by a newline, nothing else.

Here are the sentences to process:

"""

def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def save_sentences(filepath, sentences):
    with open(filepath, 'a', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence.strip() + '\n')

def post_process_sentences(text):
    """
    Post-process the output to remove any meta-commentary from the LLM
    """
    # Split into lines
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip meta-commentary lines
        if re.search(r'(^Here is|^I kept|^After processing|^These sentences|^Note:|^\d+\.|^\*)', line):
            continue
            
        # Skip lines that are instructions or explanations
        if re.search(r'(relevant because|are relevant to|removed all sentences|extracted the|have identified)', line, re.IGNORECASE):
            continue
            
        # Remove bullet points, numbers, and other formatting
        line = re.sub(r'^\s*[\*\-•]\s*', '', line)  # Remove bullet points
        line = re.sub(r'^\s*\d+\.\s*', '', line)  # Remove numbering
        
        cleaned_lines.append(line)
    
    return cleaned_lines

def clean_batch(batch_sentences, max_retries=3):
    text_batch = "\n".join(batch_sentences)
    
    for retry in range(max_retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are an agricultural research assistant specializing in wheat diseases. Return ONLY the relevant sentences about wheat diseases, treatments, and impacts. Do NOT include genetics-related content or any explanatory commentary."},
                    {"role": "user", "content": PROMPT_TEMPLATE + text_batch}
]
            )
            
            cleaned_text = response['message']['content']
            
            # Additional post-processing to filter out LLM commentary
            cleaned_sentences = post_process_sentences(cleaned_text)
            
            return cleaned_sentences
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error occurred, retrying ({retry+1}/{max_retries}): {e}")
                time.sleep(2)  # Wait a bit longer between retries
            else:
                raise

def main():
    sentences = load_sentences(INPUT_FILE)
    total_batches = (len(sentences) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Total sentences: {len(sentences)}")
    print(f"Processing {total_batches} batches using model: {OLLAMA_MODEL}...\n")
    
    total_kept = 0
    
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i+BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} sentences)...")
        
        try:
            cleaned_sentences = clean_batch(batch)
            save_sentences(OUTPUT_FILE, cleaned_sentences)
            total_kept += len(cleaned_sentences)
            keep_rate = len(cleaned_sentences) / len(batch) * 100
            print(f"  Kept {len(cleaned_sentences)}/{len(batch)} sentences ({keep_rate:.1f}%)")
            print(f"  Total kept so far: {total_kept}/{i+len(batch)} ({(total_kept/(i+len(batch))*100):.1f}%)")
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
        
        time.sleep(1)  # Short pause to avoid overwhelming Ollama
    
    print(f"\nDone! Kept {total_kept}/{len(sentences)} sentences ({(total_kept/len(sentences)*100):.1f}%)")

if __name__ == "__main__":
    main()