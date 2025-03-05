# CNN-LSTM
# Sentiment Analysis on Product Reviews using CNN-LSTM

## Overview
This project implements a **CNN-LSTM hybrid model** for **sentiment analysis** on product reviews. By combining **Convolutional Neural Networks (CNNs)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for sequential learning, this approach enhances classification accuracy and robustness in understanding customer sentiments.

## Dataset
The dataset consists of product reviews collected from e-commerce platforms. Each review includes:
- **Review Text**: The actual customer feedback.
- **Sentiment Label**: Categorized as Positive (1), Negative (0), or Neutral (2).

### Preprocessing Steps:
1. Tokenization and text cleaning (removal of special characters, stopwords, etc.).
2. Converting text into numerical features using **Word Embeddings (Word2Vec, GloVe, or FastText)**.
3. Padding sequences to ensure uniform input length for the CNN-LSTM model.

## Model Architecture
The CNN-LSTM model combines feature extraction and sequential modeling for enhanced text classification:
- **Embedding Layer**: Converts words into dense vector representations.
- **Convolutional Layer (CNN)**: Extracts spatial relationships between words, identifying key patterns.
- **Max Pooling Layer**: Reduces dimensionality while preserving important features.
- **LSTM Layer**: Captures long-term dependencies and contextual meaning in the review.
- **Fully Connected Layer**: Processes extracted features for final classification.
- **Output Layer**: Uses a softmax activation function for multi-class classification.

### Hyperparameters:
- **Embedding Dimension**: 100-300
- **CNN Filters**: 128-256
- **Kernel Size**: 3x3
- **LSTM Units**: 64-256
- **Dropout**: 0.2-0.5 (for regularization)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32 or 64
- **Epochs**: 20-50

## Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow keras nltk
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-repo/cnn-lstm-sentiment-analysis.git
cd cnn-lstm-sentiment-analysis
```
2. Run the preprocessing script:
```bash
python preprocess.py
```
3. Train the CNN-LSTM model:
```bash
python train.py
```
4. Evaluate the model:
```bash
python evaluate.py
```
5. Make predictions on new reviews:
```bash
python predict.py "The product quality is excellent!"
```

## Results
- The model achieves an accuracy of **XX%** on the test set.
- Example predictions:
  - *"I love this product!" → Positive*
  - *"The quality is terrible." → Negative*

## Future Improvements
- Implementing **Bidirectional LSTM (Bi-LSTM) and Transformer-based models** for enhanced accuracy.
- Hyperparameter tuning using **Grid Search or Bayesian Optimization**.
- Expanding the dataset with more diverse product categories.
