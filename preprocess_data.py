import os

def load_conll_data(file_path):
    """
    Loads data from a CoNLL formatted file.

    Args:
        file_path (str): The path to the CoNLL file.

    Returns:
        tuple: A tuple containing two lists:
               - sentences (list of lists of words)
               - tags (list of lists of tags)
    """
    sentences = []
    tags = []
    current_sentence = []
    current_tags = []

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "": # End of a sentence
                    if current_sentence:
                        sentences.append(current_sentence)
                        tags.append(current_tags)
                        current_sentence = []
                        current_tags = []
                else:
                    parts = line.split() # Default split handles potential tabs/spaces
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[-1] # Assume tag is the last element
                        current_sentence.append(word)
                        current_tags.append(tag)
                    else:
                        # Handle potential malformed lines, e.g., lines with only a word or tag
                        print(f"Skipping malformed line: '{line}' in file {file_path}")

            # Add the last sentence if the file doesn't end with a blank line
            if current_sentence:
                sentences.append(current_sentence)
                tags.append(current_tags)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], []

    return sentences, tags

def get_data_stats(sentences, tags, dataset_name="Dataset"):
    """Calculates and prints basic statistics about the loaded data."""
    num_sentences = len(sentences)
    num_tokens = sum(len(s) for s in sentences)
    all_tags = [tag for tag_list in tags for tag in tag_list]
    unique_tags = sorted(list(set(all_tags)))
    num_unique_tags = len(unique_tags)

    print(f"--- {dataset_name} Statistics ---")
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of unique tags: {num_unique_tags}")
    print(f"Unique tags: {unique_tags}")
    print("-" * (len(dataset_name) + 18))
    return unique_tags

if __name__ == "__main__":
    train_file = "wnut 16.txt.conll"
    test_file = "wnut 16test.txt.conll"

    print(f"Loading training data from: {train_file}")
    train_sentences, train_tags = load_conll_data(train_file)

    print(f"\nLoading test data from: {test_file}")
    test_sentences, test_tags = load_conll_data(test_file)

    if train_sentences and test_sentences:
        print("\nCalculating statistics...")
        train_unique_tags = get_data_stats(train_sentences, train_tags, "Training Set")
        test_unique_tags = get_data_stats(test_sentences, test_tags, "Test Set")

        # Check if tag sets are consistent (optional but good practice)
        if set(train_unique_tags) == set(test_unique_tags):
            print("\nTag sets are consistent between training and test data.")
        else:
            print("\nWarning: Tag sets differ between training and test data.")
            print(f"Tags only in train: {set(train_unique_tags) - set(test_unique_tags)}")
            print(f"Tags only in test: {set(test_unique_tags) - set(train_unique_tags)}")

        print("\nSample Data (First sentence):")
        if train_sentences:
            print("Train Sentence:", train_sentences[0])
            print("Train Tags:", train_tags[0])
        if test_sentences:
            print("Test Sentence:", test_sentences[0])
            print("Test Tags:", test_tags[0])

    else:
        print("\nCould not load data properly. Please check file paths and format.")
