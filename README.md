# Spam Detection NLP System

This project implements a Natural Language Processing (NLP) system to classify emails or text messages as spam or legitimate using machine learning models. It includes data preprocessing, model training, evaluation, and prediction, with support for multiple dataset sizes and models (Naive Bayes, SVM, and LSTM).

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)


## Project Overview
The system performs binary text classification to distinguish between spam and legitimate messages. It:
1. Loads a labeled dataset of emails or text messages.
2. Preprocesses text by removing stop words, lemmatizing, and cleaning special characters.
3. Trains three models: Multinomial Naive Bayes, Support Vector Machine (SVM), and Long Short-Term Memory (LSTM) neural network.
4. Evaluates models using accuracy, precision, recall, F1-score, and confusion matrices.
5. Predicts whether new messages are spam or legitimate.
6. Tests performance across different dataset sizes (e.g., 1,000 and up to the dataset's size).

The project is implemented in Python using libraries like `pandas`, `scikit-learn`, `nltk`, and `tensorflow`.

## Features
- **Data Preprocessing**: Converts text to lowercase, removes special characters, tokenizes, removes stop words, and lemmatizes.
- **Multiple Models**:
  - Naive Bayes: Fast and effective for text classification.
  - SVM: Linear kernel with probability estimates for robust performance.
  - LSTM: Deep learning model for capturing sequential patterns.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix plots.
- **Scalability Testing**: Evaluates models on varying dataset sizes (e.g., 1,000, 10,000, or dataset size).
- **Sample Prediction**: Classifies a sample message ("Win a free iPhone now! Click here to claim.") with spam probability.

## Dataset
The system uses the dataset `Spam Email raw text for NLP.csv`, which contains:
- **Columns**:
  - `CATEGORY`: Binary labels (1 for spam, 0 for legitimate).
  - `MESSAGE`: Email or text message content.
  - `FILE_NAME`: Metadata (not used in classification).
- **Size**: The dataset size is checked dynamically (e.g., ~5,572 rows based on prior runs).
- **Source**: User-provided dataset from [kaggle](https://www.kaggle.com/datasets/chandramoulinaidu/spam-classification-for-basic-nlp?resource=download). Alternatively, you can use public datasets like:
  - [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
  - [Enron-Spam Dataset](http://www.aueb.gr/users/ion/data/enron-spam/)

**Note**: Place the dataset in the `dataset/` folder as `Spam Email raw text for NLP.csv`. If using a different path or dataset, update `file_path` in `src/spam_classifier.py`.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ONEONUORA/Spam-Detection
   cd Spam-Detection
   ```

2. **Set Up a Virtual Environment (recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
The requirements.txt includes:

   ```bash
    pandas
    numpy
    scikit-learn
    nltk
    tensorflow
    matplotlib
    seaborn
   ```
For macOS with Apple Silicon, install tensorflow-macos:

    ```bash
     pip install tensorflow-macos
    ```



4. **Download NLTK Resources**:

  The script automatically downloads ```punkt```, ```stopwords```, and ```wordnet``` from nltk. If issues arise, run:

  ```bash
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```
5. **Place the Dataset**:
   
 - Copy ```Spam Email raw text for NLP.csv``` to the dataset/ folder.
 - If using a different dataset, update the file_path in  src/spam_classifier.py.

## Usage

1. **Run the Script**

  ```bash
  cd /path/to/spam-detection-nlp
  python3 src/spam_classifier.py
  ```

The script will:
- Load and preprocess the dataset.
- Train and evaluate Naive Bayes, SVM, and LSTM models for dataset sizes (e.g., 1,000 and up to the dataset size).
- Output performance metrics and confusion matrix plots.
- Predict the sample message: "Win a free iPhone now! Click here to claim."

2. Output:Console: 
- Dataset size, model performance metrics, and sample predictions.
- Summary of model performance (accuracy, precision, recall, F1-score, training time).
- Confusion matrix plots (e.g., NB_size_1000_cm.png) in the project root.

3. Example Prediction:

   ```bash
    Sample Message: Win a free iPhone now! Click here to claim.
    Prediction: Spam (Probability of Spam: 0.9946)  # SVM result
   ```

## Results
Based on a run with 1,000 samples:
- Naive Bayes:
   - Accuracy: 92.50%
   - Precision: 100.00%
   - Recall: 79.73%
   - F1-Score: 88.72%

Note: Misclassifies the sample message as "Legitimate" (probability 0.4763).

- SVM:
   - Accuracy: 98.50%
   - Precision: 100.00%
   - Recall: 95.95%
   - F1-Score: 97.93%

Correctly predicts the sample as "Spam" (probability 0.9946).

- LSTM:
   - Accuracy: 97.50%
   - Precision: 97.26%
   - Recall: 95.95%
   - F1-Score: 96.60%
Correctly predicts the sample as "Spam" (probability 0.8894).

Note: Results depend on the dataset size and content. The Naive Bayes model’s lower recall and misclassification suggest it needs tuning (e.g., smoothing or more features).



## Known Issues

1. **Dataset Size**:
   - The dataset may have fewer rows (e.g., ~5,572) than the requested 10,000 or 50,000, causing a ValueError: Cannot take a larger sample than population when 'replace=False'.
   - Fix: The script dynamically adjusts sizes to avoid this error.

2. **Naive Bayes Misclassification:** 
    - The sample message "Win a free iPhone now! Click here to claim." is misclassified as "Legitimate" by Naive Bayes.
    - Reason: Limited training data (1,000 samples) or insufficient feature extraction.

3. **LSTM Warning**:
   - The input_length parameter in the Embedding layer triggers a deprecation warning.
   - Fix: Can be removed in future updates (no impact on functionality).


## Future Improvements
1. **Naive Bayes Tuning**:
   - Add smoothing (alpha=0.5) to MultinomialNB.
   - Increase max_features or add URL detection in preprocessing.

2. **Preprocessing Enhancements**:
   - Replace URLs with a placeholder (e.g., re.sub(r'http[s]?://\S+', 'URL', text)).
   - Detect spam-specific keywords (e.g., "free", "win").

3. **Dataset Expansion**:
   - Use larger datasets like Enron-Spam or UCI SMS Spam Collection for better generalization.

4. **LSTM Optimization**:
   - Remove deprecated input_length in Embedding.
   - Experiment with more epochs or different architectures.

5. **Threshold Adjustment**:
    - Adjust the classification threshold (currently 0.6) for better spam detection.












