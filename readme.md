# Twitter Named Entity Recognition (NER) Project

This project aims to automatically identify named entities (like person, location, company) in Twitter messages using machine learning models.

## Dataset

The project uses the WNUT 16 dataset, which contains tweets annotated with 10 NER categories in CoNLL format.

*   `wnut 16.txt.conll`: Training data
*   `wnut 16test.txt.conll`: Test data

## Models Implemented

1.  **BiLSTM (Attempted with CRF):**
    *   An initial attempt was made to build a Bidirectional LSTM model with a Conditional Random Field (CRF) layer, using pre-trained Word2Vec embeddings.
    *   Due to library compatibility issues (`tensorflow-addons` deprecation, issues with alternatives like `keras-crf`), a standard BiLSTM model (without CRF) was trained as a fallback. This model (`bilstm_model.h5`) showed low performance, highlighting the difficulty without a stable CRF implementation in the environment.
2.  **BERT (Fine-tuned):**
    *   A pre-trained `bert-base-uncased` model was successfully fine-tuned for token classification using the Hugging Face `transformers` library and PyTorch.
    *   This model achieved significantly better results than the BiLSTM fallback. The final fine-tuned model and tokenizer are saved in the `./bert_ner_final_model/` directory.

## Setup

1.  **Python:** Requires Python 3.11 (due to TensorFlow version compatibility during development).
2.  **Dependencies:** Install required packages using pip:
    ```bash
    # For LSTM data prep (Word2Vec)
    pip install gensim numpy scikit-learn

    # For LSTM model training (TF 2.14 was used due to CRF issues)
    # Note: These specific versions might be needed if re-running LSTM training
    # pip install tensorflow==2.14.0 keras==2.14.0 seqeval numpy scikit-learn

    # For BERT data prep, training, and prediction
    pip install transformers torch seqeval numpy scikit-learn accelerate>=0.26.0
    ```
    *Note: It's recommended to use a virtual environment.*

## Running the Project

1.  **Prepare Data for LSTM (Optional - Low Performance):**
    ```bash
    py -3.11 train_lstm_crf.py
    ```
    This generates Word2Vec embeddings, tokenizers, and prepared data splits (`.pkl`, `.npy`, `.model` files).

2.  **Train LSTM Model (Optional - Low Performance):**
    ```bash
    py -3.11 run_lstm_crf_training.py
    ```
    This trains the BiLSTM (no CRF) model and saves it as `bilstm_model.h5`.

3.  **Prepare Data for BERT:**
    ```bash
    py -3.11 train_bert_ner.py
    ```
    This tokenizes the data using the BERT tokenizer, aligns labels, creates PyTorch datasets, and saves them (`prepared_bert_data.pkl`, `tag_map_bert.pkl`).

4.  **Train BERT Model:**
    ```bash
    py -3.11 run_bert_training.py
    ```
    This fine-tunes the `bert-base-uncased` model, evaluates it, and saves the final model and tokenizer to `./bert_ner_final_model/`. Checkpoints are saved in `./bert_ner_output/`.

5.  **Predict with BERT:**
    ```bash
    py -3.11 predict_bert.py
    ```
    This loads the fine-tuned BERT model from `./bert_ner_final_model/` and runs predictions on example sentences.

## Final Model

The best performing model (fine-tuned BERT) is saved in the `./bert_ner_final_model/` directory.
