import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import pickle
import numpy as np

# --- Configuration ---
MODEL_PATH = './bert_ner_final_model' # Path where the fine-tuned model and tokenizer are saved
TAG_MAP_PATH = "tag_map_bert.pkl"

# --- Load Model, Tokenizer, and Tag Mapping ---
print(f"Loading model and tokenizer from {MODEL_PATH}...")
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

print(f"Loading tag mapping from {TAG_MAP_PATH}...")
with open(TAG_MAP_PATH, 'rb') as f:
    tag_maps = pickle.load(f)
idx2tag = tag_maps['idx2tag']
tag2idx = tag_maps['tag2idx'] # Needed for potential checks, though idx2tag is primary for output

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --- Prediction Function ---
def predict_ner(sentence, model, tokenizer, idx2tag_map):
    """Predicts NER tags for a given sentence."""
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    # Move inputs to the correct device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model predictions
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the most likely tag index for each token
    predictions = torch.argmax(logits, dim=2)

    # Convert token IDs and prediction indices back to words and tags
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    predicted_indices = predictions[0].cpu().numpy()

    # Align tokens and predictions (ignoring special tokens and padding)
    word_tags = []
    current_word = ""
    current_tag_idx = -1 # Initialize with an invalid index

    # Use word_ids to group subword tokens
    word_ids = inputs.word_ids(batch_index=0)
    previous_word_idx = None

    for i, token in enumerate(tokens):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        word_idx = word_ids[i]
        tag_idx = predicted_indices[i]

        if word_idx is None: # Should not happen if we skip special tokens, but good check
             continue

        # If it's the start of a new word (or the first word)
        if word_idx != previous_word_idx:
            # Add the previous word and its tag if it exists
            if current_word:
                 word_tags.append((current_word, idx2tag_map.get(current_tag_idx, 'O'))) # Default to 'O'

            # Start the new word
            current_word = token
            current_tag_idx = tag_idx
        else: # It's a subword token, append to the current word
            # Remove '##' prefix if present
            current_word += token.replace('##', '')
            # Keep the tag of the first subword token (common strategy)
            # current_tag_idx remains unchanged

        previous_word_idx = word_idx

    # Add the last word
    if current_word:
         word_tags.append((current_word, idx2tag_map.get(current_tag_idx, 'O')))

    return word_tags

# --- Example Sentences ---
sentences_to_predict = [
    "Harry Potter went to London to watch the Arsenal game.",
    "Apple announced the new iPhone at the Steve Jobs Theater in Cupertino.",
    "Taylor Swift released her album 'Folklore' last year.",
    "Watching The Office on Netflix is my favorite pastime."
]

# --- Perform Predictions ---
print("\nPerforming predictions on custom sentences:")
for sentence in sentences_to_predict:
    print(f"\nSentence: {sentence}")
    predicted_tags = predict_ner(sentence, model, tokenizer, idx2tag)
    print("Predicted Tags:", predicted_tags)

print("\nPrediction complete.")
