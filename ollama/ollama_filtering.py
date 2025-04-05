import ollama
import time

# Settings
INPUT_FILE = '../extracted_corpus/combined_corpus.txt'
OUTPUT_FILE = '../extracted_corpus/ollama_filtered_corpus.txt'
BATCH_SIZE = 20
OLLAMA_MODEL = 'llama3'  # You can change this to any model you have installed

# Define the enhanced prompt
PROMPT_TEMPLATE = """
You are processing a batch of sentences extracted from agricultural research.

Your task:
- Keep only the sentences that are relevant to:
    - wheat diseases
    - wheat treatment
    - wheat yield loss
    - economic impact of wheat diseases
    - wheat disease management
    - Ignore citations, acknowledgments, website url, and content related with index

- If a sentence contains both relevant and irrelevant parts, **keep only the relevant part**.

Important context:
- The following are examples of wheat diseases: septoria, yellow rust, brown rust, stripe rust, leaf rust, glume blotch, powdery mildew, mildew, stagonospora, tan spot, ergot, fusarium head blight.
- Any sentence mentioning these diseases should be considered relevant, even if the word 'wheat' is not explicitly mentioned.
Important context:
- The following are examples of wheat diseases: septoria, yellow rust, brown rust, stripe rust, leaf rust, glume blotch, powdery mildew, mildew, stagonospora, tan spot, ergot, fusarium head blight.
- Any sentence mentioning these diseases should be considered relevant, even if the word 'wheat' is not explicitly mentioned.
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
- Sentences describing these treatments in the context of wheat should also be considered relevant.
- Sentences mentioning the **impact of wheat diseases** on:
    - **Grain yield reduction** (lower production)
    - **Grain quality loss**
    - **Financial or economic losses**
    - **Cost of disease management**
  should also be considered highly relevant and must be kept.   
- If a sentence contains both relevant and irrelevant parts, extract and keep **only** the relevant parts.

Output:
Return a list of the cleaned, relevant sentences. One per line.
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

def clean_batch(batch_sentences, max_retries=3):
    text_batch = "\n".join(batch_sentences)
    
    for retry in range(max_retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are an agricultural research assistant specializing in wheat diseases."},
                    {"role": "user", "content": PROMPT_TEMPLATE + text_batch}
                ]
            )
            cleaned_text = response['message']['content']
            cleaned_sentences = [s.strip() for s in cleaned_text.split('\n') if s.strip()]
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
