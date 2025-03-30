import pickle
import torch
from torch.utils.data import DataLoader, random_split, Dataset # Added Dataset import back
from transformers import BertForTokenClassification, TrainingArguments, Trainer, BertTokenizerFast # Removed AdamW, Added BertTokenizerFast
import numpy as np
from seqeval.metrics import classification_report, f1_score, accuracy_score

# --- Configuration ---
PREPARED_BERT_DATA_PATH = "prepared_bert_data.pkl"
MODEL_NAME = 'bert-base-uncased'
OUTPUT_DIR = './bert_ner_output' # Directory to save model checkpoints and results
LOGGING_DIR = './bert_ner_logs' # Directory for TensorBoard logs
MODEL_SAVE_PATH = './bert_ner_final_model' # Directory to save the final fine-tuned model

# Training Hyperparameters (can be tuned)
LEARNING_RATE = 3e-5
EPOCHS = 3 # Fewer epochs usually needed for fine-tuning BERT
BATCH_SIZE = 16 # Adjust based on GPU memory
WEIGHT_DECAY = 0.01
VALIDATION_SPLIT_RATIO = 0.1 # Use 10% of training data for validation

# --- Re-define NERDataset class (needed if loading from pickle) ---
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # Return tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.labels)

# --- 1. Load Processed Data ---
print("Loading processed data...")
with open(PREPARED_BERT_DATA_PATH, 'rb') as f:
    processed_data = pickle.load(f)

# The datasets are already NERDataset objects from the previous script
train_dataset_full = processed_data['train_dataset']
test_dataset = processed_data['test_dataset']
tag2idx = processed_data['tag2idx']
idx2tag = processed_data['idx2tag']
n_tags = processed_data['n_tags']

print("Data loaded successfully.")

# --- 2. Split Training Data into Train/Validation ---
print("Splitting training data into train/validation sets...")
train_size = int((1.0 - VALIDATION_SPLIT_RATIO) * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# --- 3. Load Pre-trained Model ---
print(f"\nLoading pre-trained model: {MODEL_NAME}...")
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=n_tags)

# --- 4. Define Metrics Computation ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert indices to labels
    true_labels = []
    true_predictions = []
    for prediction_list, label_list in zip(predictions, labels):
        temp_true = []
        temp_pred = []
        for prediction, label in zip(prediction_list, label_list):
            if label != -100: # Only consider non-ignored labels
                temp_true.append(idx2tag[label])
                temp_pred.append(idx2tag[prediction])
        true_labels.append(temp_true)
        true_predictions.append(temp_pred)

    # Use seqeval to compute metrics
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)

    # Extract overall metrics (micro avg is common for NER)
    results = {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
        "accuracy": accuracy_score(true_labels, true_predictions),
    }
    return results

# --- 5. Define Training Arguments ---
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOGGING_DIR,
    logging_steps=50, # Log metrics less frequently
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch", # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True, # Load the best model based on validation loss at the end
    metric_for_best_model="eval_loss", # Use validation loss to determine the best model
    greater_is_better=False, # Lower validation loss is better
    report_to="tensorboard" # Log to TensorBoard
)

# --- 6. Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- 7. Train the Model ---
print("\nStarting training...")
trainer.train()
print("Training finished.")

# --- 8. Evaluate on Test Set ---
print("\nEvaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("\nTest Set Evaluation Results:")
print(test_results)

# --- 9. Save Final Model and Tokenizer ---
print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
trainer.save_model(MODEL_SAVE_PATH)
# Tokenizer was loaded in the previous script, saving it here too for completeness
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME) # Re-load tokenizer
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Model and tokenizer saved successfully.")

print("\nBERT fine-tuning and evaluation complete.")
