# Text Classification 

## Overview

This project demonstrates the implementation of two powerful models for **text classification**:

1. **Recurrent Neural Network (RNN) using Gated Recurrent Units (GRU):**  
   - A deep learning model that utilizes GRU layers to capture temporal dependencies in sequential text data. It is designed for tasks such as sentiment analysis, topic classification, and other text classification tasks.

2. **Transformers :**  
   - A model based on the transformer architecture. This model excels at understanding contextual relationships in text and has set new standards in text classification.

## Features

- **Text Preprocessing:**  
  - Tokenization, padding, and handling of out-of-vocabulary words to prepare text data for model input.
  
- **GRU-based Model:**  
  - Implements GRU layers for capturing long-term dependencies in sequential data, designed for text classification tasks.

- **Transformer-based Model :**  
  - Utilizes text classification dataset, improving model performance.

- **Evaluation Metrics:**  
  - Performance is evaluated using accuracy, precision, recall, and F1-score to measure the effectiveness of the models.

## Requirements

- Python 3.x
- Required libraries:
  - `tensorflow` for deep learning (GRU model)
  - `transformers` for transformer models (BERT)
  - `pandas`, `numpy`, and `sklearn` for data handling and evaluation
  
Install the required libraries using pip:

```bash
pip install tensorflow transformers pandas numpy scikit-learn
