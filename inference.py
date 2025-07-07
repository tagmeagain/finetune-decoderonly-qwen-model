#!/usr/bin/env python3
"""
Inference script for the fine-tuned Qwen 1.5B model
This script loads the fine-tuned model and generates responses to user queries.
"""

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def load_fine_tuned_model(model_path="./qwen-finetuned"):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path (str): Path to the fine-tuned model
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading fine-tuned model from {model_path}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, user_query, system_prompt="", max_length=512):
    """
    Generate a response using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        user_query (str): User's question
        system_prompt (str): Optional system prompt
        max_length (int): Maximum length of generated response
    
    Returns:
        str: Generated response
    """
    # Format the input
    if system_prompt:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    return response

def main():
    """Main inference function."""
    
    # Load the fine-tuned model
    try:
        model, tokenizer = load_fine_tuned_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run the fine-tuning script first.")
        return
    
    # Sample questions to test the model
    test_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of overfitting.",
        "What are the main types of machine learning algorithms?",
        "How do you evaluate a machine learning model?",
        "What is deep learning?",
        "Explain the concept of gradient descent.",
        "What is the difference between batch gradient descent and stochastic gradient descent?",
        "How do you handle missing data in a dataset?"
    ]
    
    print("\n" + "="*50)
    print("Testing Fine-tuned Qwen 1.5B Model")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 30)
        
        try:
            response = generate_response(model, tokenizer, question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")
        
        print("-" * 50)
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main() 