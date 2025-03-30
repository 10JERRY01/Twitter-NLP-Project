# Project Documentation: Twitter Named Entity Recognition (NER)

## 1. Problem Statement

The goal of this project is to develop models capable of automatically identifying and classifying named entities within tweets. Twitter messages often contain references to people, locations, organizations, products, etc. Automatically tagging these entities can help in understanding trends, topics, and content analysis without relying solely on user-provided hashtags, which can be inconsistent or missing. This task falls under Named Entity Recognition (NER), a subtask of information extraction.

## 2. Dataset Description

*   **Source:** WNUT 16 dataset, derived from Twitter.
*   **Format:** CoNLL (Conference on Natural Language Learning) format.
    *   One word per line.
    *   Each line contains the word followed by its corresponding NER tag, separated by whitespace (tab or space).
    *   Sentences are separated by a blank line.
*   **Tagging Scheme:** BIO (Beginning, Inside, Outside) tagging scheme.
    *   `B-<TYPE>`: Marks the beginning of a named entity of a specific TYPE.
    *   `I-<TYPE>`: Marks a token inside a named entity of a specific TYPE. Must follow a `B-<TYPE>` or another `I-<TYPE>`.
    *   `O`: Marks a token that is Outside any named entity.
*   **Entity Types (10 fine-grained):** person, geo-location, company, facility, product, music artist, movie, sports team, tv show, other.

**Example (CoNLL format):**

```
Harry       B-PER
Potter      I-PER
went        O
to          O
London      B-geo-loc
.           O

Apple       B-company
announced   O
the         O
iPhone      B-product
.           O
```

## 3. Methodology

Two main approaches were implemented:

### 3.1. BiLSTM (+ CRF Attempt)

This approach uses a Bidirectional Long Short-Term Memory network, a common recurrent neural network architecture for sequence labeling tasks.

**Steps:**

1.  **Word Embeddings:** Word2Vec (Skip-gram model) was trained on the combined training and test sentences to generate 100-dimensional vector representations for words (`word2vec_lstm.model`). An embedding matrix (`embedding_matrix_lstm.npy`) was created to map words from the Keras tokenizer vocabulary to these vectors.
2.  **Data Preparation (`train_lstm_crf.py`):**
    *   Loaded CoNLL data using `load_conll_data`.
    *   Created mappings between tags and integer indices (`tag_map_lstm.pkl`).
    *   Tokenized words using `tensorflow.keras.preprocessing.text.Tokenizer` (`tokenizer_lstm.pkl`).
    *   Converted sentences and tags to sequences of integers.
    *   Padded sequences to a fixed length (`MAX_SEQ_LEN = 50`).
    *   Split the training data into training and validation sets.
    *   Saved all prepared data (`prepared_lstm_data.pkl`).
3.  **Model Architecture (`run_lstm_crf_training.py`):**
    *   Input Layer.
    *   Embedding Layer: Initialized with the pre-trained Word2Vec weights. Fine-tuning was enabled (`trainable=True`) in the final attempt.
    *   Bidirectional LSTM Layer: Processes the sequence in both forward and backward directions to capture context. Dropout was included for regularization.
    *   TimeDistributed Dense Layer: Applies a dense layer to the output of each LSTM time step.
    *   **CRF Layer (Attempted):** Initial attempts used `tensorflow_addons.layers.CRF` and later `keras_crf.CRF`. However, persistent library compatibility issues and deprecation warnings prevented successful training with a CRF layer using TensorFlow 2.14.
    *   **Fallback Output Layer:** Reverted to a `TimeDistributed(Dense(n_tags, activation='softmax'))` layer.
4.  **Training (`run_lstm_crf_training.py`):**
    *   Compiled the model using Adam optimizer and `sparse_categorical_crossentropy` loss (as CRF loss attempts failed).
    *   Used `EarlyStopping` and `ModelCheckpoint` callbacks.
    *   Trained for up to 15 epochs.
5.  **Evaluation (`run_lstm_crf_training.py`):**
    *   Used `seqeval` library for proper sequence evaluation (Precision, Recall, F1-score).
    *   The final BiLSTM model (without CRF) achieved very low performance (F1 ~0.0467), demonstrating the limitations without the sequence-aware decoding provided by a CRF layer.

### 3.2. BERT Fine-tuning

This approach leverages a pre-trained Transformer model (`bert-base-uncased`) and fine-tunes it for the specific NER task.

**Steps:**

1.  **Data Preparation (`train_bert_ner.py`):**
    *   Loaded CoNLL data.
    *   Created tag-to-index mappings (`tag_map_bert.pkl`).
    *   Loaded `BertTokenizerFast` from the `transformers` library.
    *   **Tokenization & Alignment:** Tokenized sentences using the BERT tokenizer (which uses WordPiece). Crucially, aligned the original word-level NER tags to the generated subword tokens. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) and subsequent subword tokens (tokens starting with `##`) were assigned a label of `-100` to be ignored by the loss function. Only the first token of each original word received the actual word's tag.
    *   Created PyTorch `Dataset` objects (`NERDataset`).
    *   Saved the processed datasets and mappings (`prepared_bert_data.pkl`).
2.  **Model Fine-tuning (`run_bert_training.py`):**
    *   Loaded the prepared PyTorch datasets.
    *   Split the training dataset into training and validation sets.
    *   Loaded the pre-trained `BertForTokenClassification` model, specifying the number of unique NER tags.
    *   Defined a `compute_metrics` function using `seqeval` to calculate precision, recall, F1, and accuracy during evaluation, correctly handling the `-100` ignored labels.
    *   Configured `TrainingArguments` for the Hugging Face `Trainer`, setting hyperparameters (epochs, batch size, learning rate, etc.), evaluation/saving strategies, and enabling `load_best_model_at_end`.
    *   Initialized the `Trainer`.
    *   Called `trainer.train()` to start fine-tuning.
    *   Evaluated the best model (loaded automatically by the Trainer) on the test set.
    *   Saved the final fine-tuned model and tokenizer using `trainer.save_model()` and `tokenizer.save_pretrained()` to `./bert_ner_final_model/`.
3.  **Prediction (`predict_bert.py`):**
    *   Loaded the fine-tuned model and tokenizer from the saved directory.
    *   Defined a function `predict_ner` that takes a sentence, tokenizes it, gets model predictions (logits), finds the most likely tag index for each token (`argmax`), and then aligns the predicted tags back to the original words (handling subwords by assigning the first subword's tag to the whole word).
    *   Ran predictions on example sentences.

## 4. Code Structure

*   **`preprocess_data.py`:** Contains functions to load and perform basic analysis (stats, unique tags) on the CoNLL data.
*   **`train_lstm_crf.py`:** Prepares data specifically for the LSTM model: trains Word2Vec, creates Keras tokenizer, pads sequences, creates embedding matrix, splits data, and saves artifacts.
*   **`run_lstm_crf_training.py`:** Loads LSTM-prepared data, builds the BiLSTM model (without CRF due to issues), compiles, trains (with callbacks), evaluates using `seqeval`, and saves the `.h5` model.
*   **`train_bert_ner.py`:** Prepares data specifically for the BERT model: loads CoNLL data, creates tag maps, uses `BertTokenizerFast` for tokenization and label alignment (handling subwords and special tokens), creates PyTorch `Dataset` objects, and saves them.
*   **`run_bert_training.py`:** Loads BERT-prepared data, loads the pre-trained `BertForTokenClassification` model, sets up `TrainingArguments` and `Trainer`, fine-tunes the model, evaluates it using `seqeval`, and saves the final model/tokenizer.
*   **`predict_bert.py`:** Loads the fine-tuned BERT model and tokenizer, defines a prediction function to handle tokenization and subword-to-word alignment, and runs inference on example sentences.
*   **`*.pkl`, `*.npy`, `*.model`, `*.h5`:** Data artifacts and saved models generated by the scripts.
*   **`./bert_ner_output/`:** Checkpoints saved during BERT training.
*   **`./bert_ner_logs/`:** TensorBoard logs generated during BERT training.
*   **`./bert_ner_final_model/`:** Contains the final fine-tuned BERT model (`model.safetensors` or `pytorch_model.bin`), configuration (`config.json`), tokenizer files (`tokenizer.json`, `vocab.txt`, etc.), and training arguments (`training_args.bin`).

## 5. Results

*   **BiLSTM (No CRF):** F1-score ~0.0467. Very low performance, likely due to the lack of the CRF layer to model tag dependencies and potential need for more hyperparameter tuning.
*   **BERT (Fine-tuned `bert-base-uncased`):** F1-score ~0.329. Significantly better than the BiLSTM baseline, demonstrating the effectiveness of pre-trained Transformer models for this task, even with relatively short fine-tuning (3 epochs).

## 6. Challenges Encountered

*   **CRF Layer Compatibility:** The primary challenge was integrating a CRF layer with the BiLSTM model using TensorFlow 2.14. `tensorflow-addons` is deprecated and caused errors with newer TF/Keras versions. Attempts with `keras-crf` also failed due to import/compatibility issues. This led to implementing the BiLSTM without the performance benefit of a CRF.
*   **Dependency Management:** Ensuring compatibility between TensorFlow, Keras, TensorFlow Addons (when attempted), and Python versions required downgrading TensorFlow from 2.19 to 2.14. The Hugging Face `Trainer` also required the `accelerate` library.

## 7. Potential Improvements

*   **LSTM+CRF:** Find a stable, modern CRF implementation compatible with current TensorFlow/Keras versions or implement one manually.
*   **BERT Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, number of epochs, and potentially unfreeze more layers of the BERT model during fine-tuning.
*   **Different Pre-trained Models:** Try larger BERT models (e.g., `bert-large-uncased`) or models specifically pre-trained on social media text or for NER tasks.
*   **Data Augmentation:** Augment the training data if possible.
*   **Error Analysis:** Analyze the specific errors made by the BERT model to identify patterns (e.g., confusion between certain entity types).
