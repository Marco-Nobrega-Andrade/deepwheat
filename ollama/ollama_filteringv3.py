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

OUTPUT FORMATTING:
1. **DO NOT START THE OUTPUT WITH ANY INTRODUCTION OR HEADER LIKE "HERE ARE THE RELEVANT SENTENCES" OR "RELEVANT SENTENCES:"**
2. **JUST PROVIDE EACH RELEVANT SENTENCE, ONE PER LINE, WITH ABSOLUTELY NO INTRODUCTION TEXT**
3. **DO NOT INCLUDE ANY EXPLANATIONS, TITLES, HEADERS, LABELS, BULLET POINTS, NUMBERING, OR ANY OTHER FORMATTING.**
4. **DO NOT GENERATE ANY COMMENTARY, SUMMARIES, META-TEXT, OR INSTRUCTIONS.**
5. **OUTPUT MUST BE PLAIN TEXT, EACH SENTENCE SEPARATED BY A NEWLINE, NOTHING ELSE.**
6. **DO NOT ADD ANY PREFIX LIKE "SENTENCE 1:", "RESULT:", ETC. - JUST THE PURE SENTENCES.**

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
    # Common headers that the model might insert despite instructions
    header_patterns = [
        r'^Here are the relevant sentences:',
        r'^Relevant sentences:',
        r'^The relevant sentences are:',
        r'^After filtering, the relevant sentences are:',
        r'^I have identified the following relevant sentences:',
        r'^The following sentences are relevant:',
        r'^Relevant wheat disease sentences:',
        r'^Here\'s the list of relevant sentences:',
        r'^Extracted relevant sentences:',
        r'^These sentences are relevant:',
        r'^Sentences kept:',
        r'^Selected sentences:'
    ]
    
    # First, check if the whole text starts with a header and remove it
    for pattern in header_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            # Remove the header from the beginning
            text = text[match.end():].strip()
    
    # Split into lines
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    # Skip lines if they're empty after processing
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Skip the first line if it looks like a header but wasn't caught above
        if i == 0 and any(re.search(pattern, line, re.IGNORECASE) for pattern in header_patterns):
            continue
        
        # Skip meta-commentary lines
        if re.search(r'(^Here is|^I kept|^After processing|^These sentences|^Note:|^\d+\.|^\*)', line):
            continue
        
        # Skip lines that are instructions or explanations
        if re.search(r'(relevant because|are relevant to|removed all sentences|extracted the|have identified)', line, re.IGNORECASE):
            continue
        
        # Remove bullet points, numbers, and other formatting
        line = re.sub(r'^\s*[\*\-•]\s*', '', line)  # Remove bullet points
        line = re.sub(r'^\s*\d+\.\s*', '', line)    # Remove numbering
        line = re.sub(r'^\s*[A-Z]\.\s*', '', line)  # Remove letter points (A., B., etc.)
        
        # Remove any remaining prefixes like "Sentence:", "Result:", etc.
        line = re.sub(r'^\s*(Sentence|Result|Output|Item)(\s+\d+)?:\s*', '', line, flags=re.IGNORECASE)
        
        # Skip very short lines (likely not full sentences)
        if len(line.split()) < 4:
            continue
        
        # Make sure it starts with a capital letter (likely a proper sentence)
        if not line[0].isupper():
            continue
            
        # Make sure it ends with proper punctuation
        if not line[-1] in '.!?':
            continue
        
        cleaned_lines.append(line)
    
    return cleaned_lines

def clean_batch(batch_sentences, max_retries=3):
    text_batch = "\n".join(batch_sentences)
    
    for retry in range(max_retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are an agricultural research assistant specializing in wheat diseases. IMPORTANT: Your output must contain ONLY the relevant sentences themselves with NO introduction, NO headers, NO explanations. Do NOT start your response with 'Here are...' or any similar phrase. Return ONLY the actual sentences, one per line."},
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