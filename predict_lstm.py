import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
MODEL_PATH = "bilstm_model.h5"
TOKENIZER_PATH = "tokenizer_lstm.pkl"
TAG_MAP_PATH = "tag_map_lstm.pkl"
# Load MAX_SEQ_LEN from the prepared data (or define it if known)
# We'll load it from the prepared data pickle for consistency
PREPARED_DATA_PATH = "prepared_lstm_data.pkl"

# --- Load Model, Tokenizer, Tag Map, and Config ---
print(f"Loading model from {MODEL_PATH}...")
# No custom objects needed as we reverted to standard layers
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
with open(TOKENIZER_PATH, 'rb') as f:
    word_tokenizer = pickle.load(f)
print("Tokenizer loaded successfully.")

print(f"Loading tag mapping from {TAG_MAP_PATH}...")
with open(TAG_MAP_PATH, 'rb') as f:
    tag_maps = pickle.load(f)
idx2tag = tag_maps['idx2tag']
print("Tag mapping loaded successfully.")

print(f"Loading max sequence length from {PREPARED_DATA_PATH}...")
try:
    with open(PREPARED_DATA_PATH, 'rb') as f:
        prepared_data = pickle.load(f)
    MAX_SEQ_LEN = prepared_data['max_seq_len']
    print(f"Max sequence length set to: {MAX_SEQ_LEN}")
except FileNotFoundError:
    print(f"Error: {PREPARED_DATA_PATH} not found. Using default MAX_SEQ_LEN=50.")
    MAX_SEQ_LEN = 50 # Fallback if the prepared data file is missing
except KeyError:
    print(f"Error: 'max_seq_len' key not found in {PREPARED_DATA_PATH}. Using default MAX_SEQ_LEN=50.")
    MAX_SEQ_LEN = 50 # Fallback if the key is missing

# --- Prediction Function ---
def predict_ner_lstm(sentence, model, tokenizer, idx2tag_map, max_len):
    """Predicts NER tags for a given sentence using the BiLSTM model."""
    # Tokenize the sentence words
    words = sentence.split() # Simple split for this example
    word_sequences = tokenizer.texts_to_sequences([words])

    # Pad the sequence
    padded_sequence = pad_sequences(word_sequences, maxlen=max_len, padding='post')

    # Get model predictions (probabilities)
    pred_probs = model.predict(padded_sequence, verbose=0) # verbose=0 to suppress progress bar

    # Get the tag index with the highest probability for each token
    pred_indices = np.argmax(pred_probs, axis=-1)[0] # Get the predictions for the first (only) sequence

    # Convert indices back to tags
    predicted_tags = [idx2tag_map.get(idx, 'O') for idx in pred_indices[:len(words)]] # Only map tags for original words

    # Combine words and predicted tags
    word_tags = list(zip(words, predicted_tags))

    return word_tags

# --- Example Sentences ---
sentences_to_predict = [
    "Harry Potter went to London to watch the Arsenal game.",
    "Apple announced the new iPhone at the Steve Jobs Theater in Cupertino.",
    "Taylor Swift released her album 'Folklore' last year.",
    "Watching The Office on Netflix is my favorite pastime."
]

# --- Perform Predictions ---
print("\nPerforming predictions on custom sentences (BiLSTM Model):")
for sentence in sentences_to_predict:
    print(f"\nSentence: {sentence}")
    predicted_tags = predict_ner_lstm(sentence, model, word_tokenizer, idx2tag, MAX_SEQ_LEN)
    print("Predicted Tags:", predicted_tags)

print("\nPrediction complete.")
