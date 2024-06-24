# Textual Analysis via Natural Language Processing

## Overview

This project involves two main tasks: Text Similarity and Machine Translation. Each task requires multiple setups for training and evaluation using different models and techniques.

### Task 1: Text Similarity
The objective is to calculate the similarity between two given sentences, scoring their similarity on a scale from 0 to 5.

### Task 2: Machine Translation
The aim is to translate text from German to English using various machine translation models.

## Task 1: Text Similarity

### Data Preparation
- The dataset is provided in 'train.csv' and 'dev.csv' files, which are tab-separated.
- The dataset is split into training and validation sets.

### Implementations

#### Setup 1A: BERT Model
- Train a BERT model for text similarity using the HuggingFace library.
- Obtain BERT embeddings and use an appropriate linear layer for output.
- Evaluation metric: Pearson Correlation.

#### Setup 1B: Sentence-BERT Model
- Use the Sentence-BERT model and SentenceTransformers framework.
- Encode sentences and determine cosine similarity between embeddings.
- Scale cosine similarity to match the 0 to 5 score range.

#### Setup 1C: Fine-tuned Sentence-BERT Model
- Fine-tune the Sentence-BERT model for the task of Semantic Textual Similarity (STS).
- Use CosineSimilarityLoss for training.
- Train for at least two epochs and scale cosine similarity as in Setup 1B.

## Task 2: Machine Translation

### Data Preparation
- Use the WMT 2016 dataset for translation from German to English.
- Download datasets using the HuggingFace library.

### Implementations

#### Setup 2A: Encoder-Decoder Transformer Model
- Train an encoder-decoder transformer model from scratch using PyTorch.
- Report evaluation metrics on validation and test datasets.

#### Setup 2B: Zero-Shot Evaluation with T5-Small Model
- Perform zero-shot evaluation of the t5-small model for translation.
- Generate translations and report evaluation metrics.

#### Setup 2C: Fine-tuned T5-Small Model
- Fine-tune the t5-small model for German-to-English translation.
- Train for at least two epochs with at least one layer set to trainable.
- Generate translations and report evaluation metrics.

## Evaluation Metrics
- Text Similarity: Pearson Correlation.
- Machine Translation: BLEU, METEOR, and BERTScore.
