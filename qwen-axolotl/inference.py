#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen 2.1.5 model using Axolotl
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(model_path, lora_path=None):
    """Load the base model and LoRA adapter if specified"""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load LoRA adapter if specified
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate response for a given prompt"""
    # Format prompt for Qwen 2.1.5
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
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
    
    # Extract assistant response
    if "<|im_start|>assistant\n" in response:
        assistant_response = response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]
        return assistant_response.strip()
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Inference for fine-tuned Qwen 2.1.5 model")
    parser.add_argument("--model_path", type=str, default="./outputs", 
                       help="Path to the trained model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA adapter (optional)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.lora_path)
    
    print("\n" + "="*50)
    print("Qwen 2.1.5 Fine-tuned Model Inference")
    print("="*50)
    print("Type 'quit' to exit")
    print()
    
    # Sample questions
    sample_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "What is the difference between supervised and unsupervised learning?",
        "How do you handle missing data in machine learning?",
        "What is overfitting and how do you prevent it?"
    ]
    
    print("Sample questions:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    print()
    
    # Interactive mode
    while True:
        try:
            user_input = input("Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nGenerating response...")
            response = generate_response(
                model, tokenizer, user_input, 
                max_length=args.max_length, 
                temperature=args.temperature
            )
            
            print(f"\nResponse: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 