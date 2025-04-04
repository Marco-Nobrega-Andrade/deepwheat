import re

def is_mostly_numeric_or_gibberish(sentence, threshold=0.5):
    tokens = sentence.strip().split()
    if not tokens:
        return True
    numeric_or_symbols = [t for t in tokens if re.fullmatch(r'[\W\d_]+', t)]
    return len(numeric_or_symbols) / len(tokens) > threshold

def get_avg_sentence_length(sentences):
    total_words = sum(len(s.strip().split()) for s in sentences if s.strip())
    valid_sentences = [s for s in sentences if s.strip()]
    return total_words / len(valid_sentences) if valid_sentences else 0

def remove_bad_sentences(input_file, output_file, max_words=100):
    with open(input_file, 'r', encoding='utf-8') as infile:
        sentences = infile.readlines()

    original_avg = get_avg_sentence_length(sentences)

    cleaned_sentences = []
    removed_count = 0

    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) > max_words:
            removed_count += 1
            continue
        if is_mostly_numeric_or_gibberish(sentence):
            removed_count += 1
            continue
        cleaned_sentences.append(sentence)

    cleaned_avg = get_avg_sentence_length(cleaned_sentences)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(cleaned_sentences)

    print(f"Done. Removed {removed_count} sentences.")
    print(f"Average sentence length before cleaning: {original_avg:.2f} words")
    print(f"Average sentence length after cleaning: {cleaned_avg:.2f} words")

# Example usage
remove_bad_sentences('wheat_analysis/combined_corpus.txt', 'cleaned_corpus.txt', max_words=100)