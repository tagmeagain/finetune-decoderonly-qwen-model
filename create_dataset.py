#!/usr/bin/env python3
"""
Utility script to create custom datasets for fine-tuning Qwen 1.5B
This script helps you create a properly formatted CSV dataset.
"""

import pandas as pd
import csv
import os

def create_sample_dataset():
    """Create a sample dataset with machine learning Q&A."""
    
    data = [
        # Machine Learning Basics
        {"System": "", "user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."},
        {"System": "", "user": "How does a neural network work?", "assistant": "A neural network is a computational model inspired by biological neural networks in the human brain. It consists of interconnected nodes (neurons) organized in layers. Data flows through the network, with each neuron processing inputs and passing results to the next layer. The network learns by adjusting the weights of connections between neurons based on training data."},
        {"System": "", "user": "What is the difference between supervised and unsupervised learning?", "assistant": "Supervised learning uses labeled training data to learn the relationship between inputs and outputs, allowing the model to make predictions on new data. Unsupervised learning works with unlabeled data to discover hidden patterns or structures in the data without predefined outputs."},
        
        # Deep Learning
        {"System": "", "user": "What is deep learning?", "assistant": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition."},
        {"System": "", "user": "What are the main types of neural networks?", "assistant": "The main types of neural networks include: 1) Feedforward Neural Networks (FNN) - basic neural networks, 2) Convolutional Neural Networks (CNN) - for image processing, 3) Recurrent Neural Networks (RNN) - for sequential data, 4) Long Short-Term Memory (LSTM) - for long sequences, 5) Transformer Networks - for attention-based processing."},
        
        # Model Training
        {"System": "", "user": "What is overfitting in machine learning?", "assistant": "Overfitting occurs when a machine learning model learns the training data too well, including noise and irrelevant patterns, which leads to poor performance on new, unseen data. This happens when the model becomes too complex relative to the amount of training data available."},
        {"System": "", "user": "How do you prevent overfitting?", "assistant": "Overfitting can be prevented through several techniques: 1) Regularization (L1/L2), 2) Dropout layers, 3) Early stopping, 4) Cross-validation, 5) Data augmentation, 6) Reducing model complexity, 7) Increasing training data, and 8) Ensemble methods."},
        {"System": "", "user": "What is gradient descent?", "assistant": "Gradient descent is an optimization algorithm used to minimize the loss function of a machine learning model. It works by iteratively adjusting the model's parameters in the direction of the steepest descent of the loss function, gradually moving toward the minimum point."},
        
        # Data Processing
        {"System": "", "user": "How do you handle missing data in a dataset?", "assistant": "Missing data can be handled through several methods: 1) Deletion (removing rows or columns with missing values), 2) Imputation (filling missing values with mean, median, mode, or predicted values), 3) Using algorithms that handle missing data natively, or 4) Creating a separate category for missing values."},
        {"System": "", "user": "What is data normalization?", "assistant": "Data normalization is the process of scaling features to a standard range, typically between 0 and 1 or -1 and 1. This helps ensure that all features contribute equally to the model and prevents features with larger scales from dominating the learning process."},
        
        # Evaluation
        {"System": "", "user": "How do you evaluate a machine learning model?", "assistant": "Machine learning models are evaluated using various metrics depending on the task. For classification: accuracy, precision, recall, F1-score, and ROC-AUC. For regression: mean squared error, mean absolute error, and R-squared. Cross-validation and train/test splits help ensure reliable evaluation."},
        {"System": "", "user": "What is cross-validation?", "assistant": "Cross-validation is a technique for assessing how well a model will generalize to new data. It involves splitting the data into multiple folds, training the model on some folds and testing on others, then averaging the results. This provides a more robust estimate of model performance."},
        
        # Applications
        {"System": "", "user": "What are some real-world applications of machine learning?", "assistant": "Machine learning has numerous real-world applications: 1) Recommendation systems (Netflix, Amazon), 2) Image and speech recognition, 3) Natural language processing, 4) Fraud detection, 5) Medical diagnosis, 6) Autonomous vehicles, 7) Financial forecasting, 8) Customer service chatbots, and 9) Drug discovery."},
        {"System": "", "user": "What is natural language processing?", "assistant": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It includes tasks like text classification, sentiment analysis, machine translation, question answering, and text generation. Modern NLP often uses transformer-based models like BERT and GPT."},
        
        # Tools and Frameworks
        {"System": "", "user": "What are popular machine learning frameworks?", "assistant": "Popular machine learning frameworks include: 1) TensorFlow - Google's framework, 2) PyTorch - Facebook's framework, 3) Scikit-learn - for traditional ML, 4) Keras - high-level API, 5) XGBoost - for gradient boosting, 6) LightGBM - Microsoft's gradient boosting, and 7) Hugging Face Transformers - for NLP models."},
        {"System": "", "user": "What is the difference between TensorFlow and PyTorch?", "assistant": "TensorFlow and PyTorch are both popular deep learning frameworks. TensorFlow offers better production deployment and mobile support, while PyTorch is more Pythonic and flexible for research. TensorFlow has TensorBoard for visualization, while PyTorch integrates well with Python's scientific computing ecosystem. Both support dynamic and static computation graphs."},
    ]
    
    return pd.DataFrame(data)

def create_custom_dataset():
    """Interactive function to create a custom dataset."""
    
    print("Creating Custom Dataset")
    print("=" * 40)
    print("Enter your Q&A pairs. Type 'done' when finished.")
    print("Format: Question -> Answer")
    print()
    
    data = []
    
    while True:
        question = input("Question (or 'done' to finish): ").strip()
        
        if question.lower() == 'done':
            break
        
        if not question:
            print("Question cannot be empty. Please try again.")
            continue
        
        answer = input("Answer: ").strip()
        
        if not answer:
            print("Answer cannot be empty. Please try again.")
            continue
        
        # Ask for system prompt (optional)
        system_prompt = input("System prompt (optional, press Enter to skip): ").strip()
        
        data.append({
            "System": system_prompt,
            "user": question,
            "assistant": answer
        })
        
        print(f"Added: {question[:50]}...")
        print()
    
    if data:
        return pd.DataFrame(data)
    else:
        print("No data entered. Creating empty dataset.")
        return pd.DataFrame(columns=["System", "user", "assistant"])

def save_dataset(df, filename="custom_dataset.csv"):
    """Save dataset to CSV file."""
    
    try:
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        print(f"Dataset saved to {filename}")
        print(f"Total entries: {len(df)}")
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False

def validate_dataset(df):
    """Validate the dataset format."""
    
    required_columns = ["System", "user", "assistant"]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for empty rows
    empty_rows = df[df['user'].isna() | df['assistant'].isna()].shape[0]
    if empty_rows > 0:
        print(f"Warning: {empty_rows} rows have empty user or assistant fields")
    
    # Check dataset size
    if len(df) == 0:
        print("Warning: Dataset is empty")
        return False
    
    print(f"Dataset validation passed!")
    print(f"Total entries: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return True

def main():
    """Main function to create datasets."""
    
    print("Qwen 1.5B Dataset Creator")
    print("=" * 30)
    print()
    
    while True:
        print("Choose an option:")
        print("1. Create sample dataset (machine learning Q&A)")
        print("2. Create custom dataset (interactive)")
        print("3. Validate existing dataset")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nCreating sample dataset...")
            df = create_sample_dataset()
            filename = input("Enter filename (default: sample_dataset.csv): ").strip()
            if not filename:
                filename = "sample_dataset.csv"
            save_dataset(df, filename)
            
        elif choice == "2":
            print("\nCreating custom dataset...")
            df = create_custom_dataset()
            if len(df) > 0:
                filename = input("Enter filename (default: custom_dataset.csv): ").strip()
                if not filename:
                    filename = "custom_dataset.csv"
                save_dataset(df, filename)
            
        elif choice == "3":
            filename = input("Enter dataset filename: ").strip()
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    validate_dataset(df)
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print(f"File {filename} not found")
                
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main() 