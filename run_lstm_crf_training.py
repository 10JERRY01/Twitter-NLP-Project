import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Activation
# Note: CRF layer attempts using tensorflow-addons and keras-crf failed due to library incompatibilities/deprecation.
# This script now trains a standard BiLSTM model without a CRF layer.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from seqeval.metrics import classification_report, f1_score, accuracy_score # Added accuracy_score back for consistency

# --- Configuration ---
PREPARED_DATA_PATH = "prepared_lstm_data.pkl"
EMBEDDING_MATRIX_PATH = "embedding_matrix_lstm.npy"
TAG_MAP_PATH = "tag_map_lstm.pkl"
MODEL_SAVE_PATH = "bilstm_model.h5" # Changed to .h5 format

# Hyperparameters (can be tuned later)
LSTM_UNITS = 64
DROPOUT_RATE = 0.1 # Added dropout for regularization
LEARNING_RATE = 0.001
EPOCHS = 15 # Increased epochs, will use EarlyStopping
BATCH_SIZE = 32
PATIENCE = 3 # For EarlyStopping

# --- 1. Load Prepared Data and Artifacts ---
print("Loading prepared data and artifacts...")
with open(PREPARED_DATA_PATH, 'rb') as f:
    prepared_data = pickle.load(f)

X_train = prepared_data['X_train']
y_train = prepared_data['y_train']
X_val = prepared_data['X_val']
y_val = prepared_data['y_val']
X_test = prepared_data['X_test']
y_test = prepared_data['y_test']
vocab_size = prepared_data['vocab_size']
n_tags = prepared_data['n_tags']
max_seq_len = prepared_data['max_seq_len']

embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)
embedding_dim = embedding_matrix.shape[1] # Get embedding dim from matrix

with open(TAG_MAP_PATH, 'rb') as f:
    tag_maps = pickle.load(f)
idx2tag = tag_maps['idx2tag']

print("Data loaded successfully.")
print(f"Vocab size: {vocab_size}, Embedding dim: {embedding_dim}, Max seq len: {max_seq_len}, Num tags: {n_tags}")

# --- 2. Build BiLSTM Model ---
print("\nBuilding BiLSTM model (CRF attempts failed due to library issues)...")

# Input layer
input_layer = Input(shape=(max_seq_len,))

# Embedding layer (using pre-trained Word2Vec weights)
# Set trainable=False initially, can be fine-tuned later if needed
embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            mask_zero=True, # Important for handling padding
                            trainable=True)(input_layer) # Allow fine-tuning embeddings

# Bidirectional LSTM layer
# return_sequences=True is crucial for sequence labeling
bilstm_layer = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))(embedding_layer)

# TimeDistributed Dense layer
# TimeDistributed Dense layer with softmax activation for standard sequence classification
time_distributed_dense = TimeDistributed(Dense(n_tags))(bilstm_layer)
output_layer = Activation('softmax')(time_distributed_dense)

# Define the model
model = Model(input_layer, output_layer)
model.summary()

# --- 3. Compile Model ---
print("\nCompiling model...")
# Use the CRF potential function as the loss
# The CRF layer handles the decoding and loss calculation internally when used as the output layer.
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Compile with sparse_categorical_crossentropy loss and accuracy metric.
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# --- 4. Train Model ---
print("\nTraining model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)

print("Training finished.")
print(f"Best model saved to {MODEL_SAVE_PATH}")

# --- 5. Evaluate Model ---
print("\nEvaluating model on test set...")

# Predict probabilities for each tag at each time step
y_pred_probs = model.predict(X_test)
# Get the tag index with the highest probability at each step
y_pred_indices = np.argmax(y_pred_probs, axis=-1)

# Convert indices back to tags, ignoring padding (where X_test is 0)
y_true_tags = []
y_pred_tags = []

for i in range(len(X_test)):
    true_seq = []
    pred_seq = []
    for j in range(max_seq_len):
        if X_test[i, j] != 0: # Check if it's not a padding token
            true_tag_idx = y_test[i, j]
            pred_tag_idx = y_pred_indices[i, j]
            true_seq.append(idx2tag[true_tag_idx])
            pred_seq.append(idx2tag[pred_tag_idx])
    y_true_tags.append(true_seq)
    y_pred_tags.append(pred_seq)

# Calculate and print classification report
report = classification_report(y_true_tags, y_pred_tags, digits=4)
print("\nClassification Report (Test Set):")
print(report)

# Calculate overall F1 score (micro average is often used for NER)
f1 = f1_score(y_true_tags, y_pred_tags, average='micro')
print(f"\nOverall F1 Score (Micro): {f1:.4f}")

print("\nEvaluation complete.")
