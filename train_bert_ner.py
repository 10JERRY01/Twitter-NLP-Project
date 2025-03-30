import os
import pickle
import numpy as np
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch

# --- Data Loading Function (copied from preprocess_data.py) ---
def load_conll_data(file_path):
    """Loads data from a CoNLL formatted file."""
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
                if line == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        tags.append(current_tags)
                        current_sentence = []
                        current_tags = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[-1]
                        current_sentence.append(word)
                        current_tags.append(tag)
                    else:
                        print(f"Skipping malformed line: '{line}' in file {file_path}")
            if current_sentence:
                sentences.append(current_sentence)
                tags.append(current_tags)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], []
    return sentences, tags
# --- End of Data Loading Function ---

# --- Configuration ---
TRAIN_FILE = "wnut 16.txt.conll"
TEST_FILE = "wnut 16test.txt.conll"
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128 # Max sequence length for BERT
BATCH_SIZE = 16 # Adjust based on GPU memory
TAG_MAP_PATH = "tag_map_bert.pkl" # Separate tag map for BERT potentially
PREPARED_BERT_DATA_PATH = "prepared_bert_data.pkl"

# --- 1. Load Data ---
print("Loading data...")
train_sentences, train_tags = load_conll_data(TRAIN_FILE)
test_sentences, test_tags = load_conll_data(TEST_FILE) # Using test set for now, will split train later

if not train_sentences or not test_sentences:
    print("Failed to load data. Exiting.")
    exit()

print(f"Loaded {len(train_sentences)} training sentences and {len(test_sentences)} test sentences.")

# --- 2. Create Tag Mapping ---
print("\nCreating tag mapping...")
# Use tags from training data only to define the mapping
all_tags_flat_train = [tag for sublist in train_tags for tag in sublist]
unique_tags = sorted(list(set(all_tags_flat_train)))
tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
idx2tag = {i: tag for tag, i in tag2idx.items()}
n_tags = len(unique_tags)
print(f"Number of unique tags (from train set): {n_tags}")
print(f"Tag mapping: {tag2idx}")

# Save tag mapping
with open(TAG_MAP_PATH, 'wb') as f:
    pickle.dump({'tag2idx': tag2idx, 'idx2tag': idx2tag, 'n_tags': n_tags}, f)
print(f"Tag mapping saved to {TAG_MAP_PATH}")

# --- 3. Tokenization and Label Alignment ---
print(f"\nLoading tokenizer: {MODEL_NAME}...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(sentences, tags, tokenizer, tag2idx_map):
    """
    Tokenizes sentences using the provided tokenizer and aligns the
    corresponding NER tags to the generated tokens (WordPieces/subwords).
    Uses -100 for special tokens and subsequent subword tokens,
    so they are ignored by the loss function during training.
    """
    # Ensure 'O' tag exists for default assignment
    o_tag_idx = tag2idx_map.get('O')
    if o_tag_idx is None:
        # This should ideally not happen if 'O' is in the training data,
        # but handle it just in case. Assign a default index (e.g., 0)
        # or raise an error. Here, we'll print a warning and use 0.
        print("Warning: 'O' tag not found in tag2idx mapping. Using index 0 as default.")
        o_tag_idx = 0 # Or choose another appropriate default/error handling

    tokenized_inputs = tokenizer(sentences, truncation=True, is_split_into_words=True, padding='max_length', max_length=MAX_LEN)
    labels = []
    for i, label_list in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: # Special tokens like [CLS], [SEP]
                label_ids.append(-100) # Ignore special tokens in loss calculation
            elif word_idx != previous_word_idx: # First token of a new word
                # Assign the label of the current word
                # Use .get() with default 'O' tag index for robustness
                label_ids.append(tag2idx_map.get(label_list[word_idx], o_tag_idx))
            else: # Subsequent tokens of the same word
                # Assign -100 to ignore these subword tokens in loss calculation
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing and aligning labels for training set...")
train_encodings = tokenize_and_align_labels(train_sentences, train_tags, tokenizer, tag2idx)
print("Tokenizing and aligning labels for test set...")
test_encodings = tokenize_and_align_labels(test_sentences, test_tags, tokenizer, tag2idx)

# --- 4. Create PyTorch Dataset ---
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Return tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.labels)

train_dataset = NERDataset(train_encodings)
test_dataset = NERDataset(test_encodings) # Will split train_dataset later for validation

print(f"\nSample Train Encoding (Input IDs): {train_dataset[0]['input_ids']}")
print(f"Sample Train Encoding (Labels): {train_dataset[0]['labels']}")

# --- 5. Save Processed Data ---
# Save datasets and tokenizer info for the training script
processed_data = {
    'train_dataset': train_dataset,
    'test_dataset': test_dataset, # Note: This is the original test set
    'tag2idx': tag2idx,
    'idx2tag': idx2tag,
    'n_tags': n_tags
}
with open(PREPARED_BERT_DATA_PATH, 'wb') as f:
    pickle.dump(processed_data, f)
print(f"\nProcessed data saved to {PREPARED_BERT_DATA_PATH}")

print("\nBERT data preparation finished.")
