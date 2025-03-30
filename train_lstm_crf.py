import os
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle # To save tokenizer and mappings

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
W2V_MODEL_PATH = "word2vec_lstm.model"
TOKENIZER_PATH = "tokenizer_lstm.pkl"
TAG_MAP_PATH = "tag_map_lstm.pkl"
EMBEDDING_MATRIX_PATH = "embedding_matrix_lstm.npy" # Added path for saving matrix
PREPARED_DATA_PATH = "prepared_lstm_data.pkl" # Added path for saving prepared data

EMBEDDING_DIM = 100  # Dimension for Word2Vec and Embedding layer
MAX_SEQ_LEN = 50     # Maximum sequence length after padding
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# --- 1. Load Data ---
print("Loading data...")
train_sentences, train_tags = load_conll_data(TRAIN_FILE)
test_sentences, test_tags = load_conll_data(TEST_FILE)

if not train_sentences or not test_sentences:
    print("Failed to load data. Exiting.")
    exit()

print(f"Loaded {len(train_sentences)} training sentences and {len(test_sentences)} test sentences.")

# --- 2. Train Word2Vec ---
print("\nTraining Word2Vec model...")
# Combine train and test sentences for a richer vocabulary
all_sentences = train_sentences + test_sentences
w2v_model = Word2Vec(sentences=all_sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4, sg=1) # Using Skip-gram
w2v_model.save(W2V_MODEL_PATH)
print(f"Word2Vec model saved to {W2V_MODEL_PATH}")
print(f"Vocabulary size: {len(w2v_model.wv.index_to_key)}")

# --- 3. Prepare Data for TensorFlow ---
print("\nPreparing data for TensorFlow...")

# 3.1. Create Tag Mapping
all_tags_flat = [tag for sublist in train_tags + test_tags for tag in sublist]
unique_tags = sorted(list(set(all_tags_flat)))
tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
idx2tag = {i: tag for tag, i in tag2idx.items()}
n_tags = len(unique_tags)
print(f"Number of unique tags: {n_tags}")
print(f"Tag mapping: {tag2idx}")

# Save tag mapping
with open(TAG_MAP_PATH, 'wb') as f:
    pickle.dump({'tag2idx': tag2idx, 'idx2tag': idx2tag}, f)
print(f"Tag mapping saved to {TAG_MAP_PATH}")

# 3.2. Tokenize Words
# Use Keras Tokenizer, fit on training sentences only to avoid data leakage
word_tokenizer = Tokenizer(oov_token="<OOV>") # Out-of-vocabulary token
word_tokenizer.fit_on_texts(train_sentences)
vocab_size = len(word_tokenizer.word_index) + 1 # +1 for padding token 0
print(f"Vocabulary size (Keras Tokenizer): {vocab_size}")

# Save tokenizer
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(word_tokenizer, f)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

# 3.3. Convert Sentences and Tags to Sequences
X_train = word_tokenizer.texts_to_sequences(train_sentences)
y_train = [[tag2idx[tag] for tag in tags] for tags in train_tags]

X_test = word_tokenizer.texts_to_sequences(test_sentences)
y_test = [[tag2idx[tag] for tag in tags] for tags in test_tags]

# 3.4. Pad Sequences
print(f"\nPadding sequences to max length: {MAX_SEQ_LEN}...")
X_train_padded = pad_sequences(X_train, maxlen=MAX_SEQ_LEN, padding='post')
y_train_padded = pad_sequences(y_train, maxlen=MAX_SEQ_LEN, padding='post', value=tag2idx['O']) # Pad tags with 'O' tag index

X_test_padded = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding='post')
y_test_padded = pad_sequences(y_test, maxlen=MAX_SEQ_LEN, padding='post', value=tag2idx['O'])

print("Padding complete.")

# --- 4. Train/Validation Split ---
print("\nSplitting data into training and validation sets...")
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_padded, y_train_padded, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
)

print(f"Training sequences shape: {X_train_split.shape}")
print(f"Training tags shape: {y_train_split.shape}")
print(f"Validation sequences shape: {X_val_split.shape}")
print(f"Validation tags shape: {y_val_split.shape}")
print(f"Test sequences shape: {X_test_padded.shape}")
print(f"Test tags shape: {y_test_padded.shape}")

# --- 5. Create Embedding Matrix (using Word2Vec) ---
print("\nCreating embedding matrix...")
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
hits = 0
misses = 0
for word, i in word_tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
        hits += 1
    else:
        misses += 1
        # Words not found in Word2Vec will be initialized to zero vectors.

print(f"Converted {hits} words ({misses} misses)")
print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Save embedding matrix
np.save(EMBEDDING_MATRIX_PATH, embedding_matrix)
print(f"Embedding matrix saved to {EMBEDDING_MATRIX_PATH}")

# Save prepared data splits for later use in model training script
prepared_data = {
    'X_train': X_train_split,
    'y_train': y_train_split,
    'X_val': X_val_split,
    'y_val': y_val_split,
    'X_test': X_test_padded,
    'y_test': y_test_padded,
    'vocab_size': vocab_size,
    'n_tags': n_tags,
    'max_seq_len': MAX_SEQ_LEN
}
with open(PREPARED_DATA_PATH, 'wb') as f:
    pickle.dump(prepared_data, f)
print(f"Prepared data saved to {PREPARED_DATA_PATH}")


print("\nData preparation finished. Ready for model building and training.")
