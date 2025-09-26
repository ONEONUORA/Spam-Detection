
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Download NLTK resources for tokenization, stop words, and lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Data Loading
def load_data(file_path, size=None):
    """
    Loads the spam  dataset from a CSV file and extracts message text and labels.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        size (int, optional): Number of samples to randomly select. If None, uses the entire dataset.
    Returns:
        tuple: Two arrays containing the messages (X) and binary labels (y, 1 for spam, 0 for legit).
    Raises:
        ValueError: If the 'CATEGORY' column has an unexpected format.
    """
    df = pd.read_csv(file_path)
    print("Columns:", df.columns.tolist())  # Debug: Display column names
    print("First few rows:\n", df.head())   # Debug: Show sample data
    print("Unique CATEGORY values:", df['CATEGORY'].unique())  # Debug: Show unique label values
    if size:
        df = df.sample(n=size, random_state=42)  # Randomly sample specified number of rows
    # Convert labels to binary (1 for spam, 0 for legit)
    if df['CATEGORY'].dtype == object:  # Handle string labels (e.g., 'spam'/'ham')
        df['CATEGORY'] = df['CATEGORY'].map({'spam': 1, 'ham': 0, 'Spam': 1, 'Ham': 0})
    elif df['CATEGORY'].dtype in ['int64', 'int32']:  # Handle numeric labels (1/0)
        pass
    else:
        raise ValueError("Unexpected CATEGORY format. Please check unique values.")
    return df['MESSAGE'].values, df['CATEGORY'].values

# 2. Preprocessing
def preprocess_text(texts):
    """
    Preprocesses a list of text messages by cleaning, tokenizing, removing stop words, and lemmatizing.
    Args:
        texts (list): List of text strings to preprocess.
    Returns:
        list: Preprocessed text strings.
    """
    stop_words = set(stopwords.words('english'))  # Load English stop words
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    
    def clean_text(text):
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters, keeping only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize text into words
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize remaining words
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)  # Join tokens back into a single string
    
    return [clean_text(text) for text in texts]  # Apply cleaning to all texts

# 3. Model Training and Evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='nb'):
    """
    Trains a specified machine learning model and evaluates its performance on test data.
    Args:
        X_train (list): Training text data.
        X_test (list): Test text data.
        y_train (array): Training labels (1 for spam, 0 for legit).
        y_test (array): Test labels.
        model_type (str): Type of model to train ('nb', 'svm', or 'lstm'). Defaults to 'nb'.
    Returns:
        tuple: Trained model, TF-IDF vectorizer, tokenizer (for LSTM), max sequence length (for LSTM),
               training time, and predicted test labels.
    """
    vectorizer = TfidfVectorizer(max_features=10000)  # Convert text to TF-IDF features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    start_time = time.time()  # Record training start time
    if model_type == 'nb':
        model = MultinomialNB()  # Initialize Naive Bayes classifier
        model.fit(X_train_tfidf, y_train)  # Train on TF-IDF features
    elif model_type == 'svm':
        model = SVC(kernel='linear', probability=True)  # Initialize SVM with probability estimates
        model.fit(X_train_tfidf, y_train)  # Train on TF-IDF features
    elif model_type == 'lstm':
        vocab_size = 5000  # Vocabulary size for tokenization
        max_length = 100  # Maximum sequence length
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(X_train)  # Fit tokenizer on training text
        X_train_seq = tokenizer.texts_to_sequences(X_train)  # Convert text to sequences
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_length)  # Pad sequences
        X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_length)
        
        model = Sequential([  # Build LSTM model
            Embedding(vocab_size, 128, input_length=max_length),
            SpatialDropout1D(0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)  # Train LSTM
        y_pred = (model.predict(X_test_pad) > 0.5).astype(int)  # Predict binary labels
        training_time = time.time() - start_time
        return model, vectorizer, tokenizer, max_length, training_time, y_pred.flatten()
    
    y_pred = model.predict(X_test_tfidf)  # Predict for NB or SVM
    training_time = time.time() - start_time  # Calculate training time
    
    return model, vectorizer, None, None, training_time, y_pred

# 4. Evaluate Model
def evaluate_model(y_test, y_pred, model_name):
    """
    Evaluates a model's performance using accuracy, precision, recall, F1-score, and confusion matrix.
    Args:
        y_test (array): True test labels.
        y_pred (array): Predicted test labels.
        model_name (str): Name of the model for display purposes.
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    precision = precision_score(y_test, y_pred)  # Calculate precision
    recall = recall_score(y_test, y_pred)  # Calculate recall
    f1 = f1_score(y_test, y_pred)  # Calculate F1-score
    cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_cm.png')
    plt.close()
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# 5. Predict New Messages
def predict_new_message(message, model, vectorizer, tokenizer=None, max_length=None, model_type='nb'):
    """
    Classifies a new message as spam or legitimate using a trained model.
    Args:
        message (str): The message to classify.
        model: Trained model (Naive Bayes, SVM, or LSTM).
        vectorizer: TF-IDF vectorizer for NB/SVM.
        tokenizer: Tokenizer for LSTM.
        max_length: Maximum sequence length for LSTM.
        model_type (str): Type of model ('nb', 'svm', 'lstm'). Defaults to 'nb'.
    Returns:
        tuple: Predicted label ('Spam' or 'Legitimate') and spam probability.
    """
    processed = preprocess_text([message])[0]  # Preprocess the input message
    if model_type == 'lstm':
        seq = tokenizer.texts_to_sequences([processed])  # Convert to sequence
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length)  # Pad sequence
        pred = model.predict(pad)[0][0]  # Predict probability
    else:
        vec = vectorizer.transform([processed])  # Convert to TF-IDF
        if model_type == 'svm':
            try:
                pred = model.predict_proba(vec)[0][1]  # Try probability prediction
            except AttributeError:
                pred = model.decision_function(vec)[0]  # Fallback to decision function
                pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize to [0,1]
        else:
            pred = model.predict_proba(vec)[0][1]  # Naive Bayes probability
    return 'Spam' if pred > 0.6 else 'Legitimate', pred  # Classify based on threshold

# 6. Main Function
def main():
    """
    Orchestrates the entire spam detection pipeline: loads data, preprocesses, trains models,
    evaluates performance, and predicts on a sample message. Saves results to a report.
    """
    file_path = 'dataset/Spam Email raw text for NLP.csv'  # Path to dataset
    
    # Check dataset size
    df = pd.read_csv(file_path)
    dataset_size = df.shape[0]
    print(f"Total dataset size: {dataset_size} rows")
    
    # Adjust sizes based on dataset size
    sizes = [1000, min(10000, dataset_size), min(50000, dataset_size)]
    sizes = sorted(list(set(sizes)))  # Remove duplicates and sort
    models = ['nb', 'svm', 'lstm']
    results = {}
    
    for size in sizes:
        print(f"\nTesting with dataset size: {size}")
        X, y = load_data(file_path, size)  # Load data
        X_processed = preprocess_text(X)  # Preprocess text
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)  # Split data
        
        for model_type in models:
            print(f"\nTraining {model_type.upper()}...")
            model, vectorizer, tokenizer, max_length, training_time, y_pred = train_and_evaluate(
                X_train, X_test, y_train, y_test, model_type)  # Train and evaluate
            result = evaluate_model(y_test, y_pred, f"{model_type.upper()}_size_{size}")  # Evaluate model
            result['training_time'] = training_time
            results[f"{model_type}_size_{size}"] = result
            
            # Example prediction
            sample_message = "Win a free iPhone now! Click here to claim."
            label, prob = predict_new_message(sample_message, model, vectorizer, tokenizer, max_length, model_type)
            print(f"Sample Message: {sample_message}")
            print(f"Prediction: {label} (Probability of Spam: {prob:.4f})")
    
    # Save results to report
    with open('../report.md', 'w') as f:
        f.write("# Spam Detection System Report\n\n")
        for key, metrics in results.items():
            f.write(f"## {key}\n")
            f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"- Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['recall']:.4f}\n")
            f.write(f"- F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"- Training Time: {metrics['training_time']:.2f} seconds\n\n")

if __name__ == "__main__":
    main()