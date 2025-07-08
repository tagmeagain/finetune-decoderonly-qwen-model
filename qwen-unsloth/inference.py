#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen model using Unsloth
"""

import torch
from unsloth import FastLanguageModel
import argparse

def load_model(model_path):
    """Load the fine-tuned model"""
    print(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_new_tokens=512, temperature=0.7):
    """Generate response for a given instruction"""
    # Create prompt in the same Alpaca format as training
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    
    prompt = alpaca_prompt.format(instruction, input_text)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part
    if "### Response:\n" in response:
        response = response.split("### Response:\n")[-1].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Qwen model")
    parser.add_argument("--model_path", type=str, default="./outputs", 
                       help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    print("\n" + "="*60)
    print("🤖 Qwen Fine-tuned Model Inference")
    print("="*60)
    print("Type 'quit' to exit")
    print()
    
    # Sample questions
    sample_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "What is the difference between supervised and unsupervised learning?",
        "How do you handle missing data in machine learning?",
        "What is overfitting and how do you prevent it?",
        "Explain the concept of cross-validation.",
        "What is the bias-variance tradeoff?",
        "How does gradient descent work?",
        "What is the difference between precision and recall?",
        "Explain the concept of ensemble methods."
    ]
    
    print("📚 Sample questions:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i:2d}. {question}")
    print()
    
    # Interactive mode
    while True:
        try:
            user_input = input("💬 Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🤔 Generating response...")
            response = generate_response(
                model, tokenizer, user_input,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            print(f"\n🤖 Response: {response}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 