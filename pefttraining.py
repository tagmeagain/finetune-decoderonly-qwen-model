#!/usr/bin/env python3
"""
Qwen2 1.5B LoRA Fine-tuning Script - Function-based approach
Supports CSV/JSON datasets with 'input' and 'output' columns
"""

import os
import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import argparse
from typing import Dict, List

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from CSV or JSON file"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

def format_prompt(input_text: str, output_text: str) -> str:
    """Format input and output into conversation format"""
    return f"Human: {input_text}\n\nAssistant: {output_text}"

def tokenize_function(examples, tokenizer, max_length: int = 512):
    """Tokenize the examples for training"""
    prompts = [format_prompt(inp, out) for inp, out in zip(examples['input'], examples['output'])]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized

def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2-1.5B"):
    """Setup model and tokenizer with LoRA configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(data: List[Dict], tokenizer, max_length: int = 512):
    """Prepare dataset for training"""
    
    # Validate dataset format
    if not all('input' in item and 'output' in item for item in data):
        raise ValueError("Dataset must contain 'input' and 'output' columns")
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_list(data)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def create_training_arguments(args):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        fp16=True,  # Use mixed precision
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory
        save_total_limit=2,  # Keep only 2 checkpoints
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )

def train_model(model, tokenizer, train_dataset, training_args):
    """Train the model"""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    return trainer

def save_model(trainer, tokenizer, output_dir: str):
    """Save the trained model and tokenizer"""
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 1.5B with LoRA")
    parser.add_argument("--dataset", required=True, help="Path to dataset file (CSV/JSON)")
    parser.add_argument("--output_dir", default="./qwen2-lora-finetuned", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} samples")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset = prepare_dataset(data, tokenizer, args.max_length)
    
    # Create training arguments
    training_args = create_training_arguments(args)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, training_args)
    
    # Save model
    save_model(trainer, tokenizer, args.output_dir)
    
    print("Training completed successfully!")

def test_model(model_path: str, prompt: str):
    """Test the fine-tuned model"""
    from peft import PeftModel
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Format prompt
    formatted_prompt = f"Human: {prompt}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response:\n{response}")

def create_sample_dataset(filename: str = "sample_dataset.csv"):
    """Create a sample dataset for testing"""
    sample_data = [
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"input": "Explain what machine learning is", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"input": "What is photosynthesis?", "output": "Photosynthesis is the process by which plants convert light energy, usually from the sun, into chemical energy stored in glucose."},
        {"input": "How do you make coffee?", "output": "To make coffee, you need ground coffee beans, hot water, and a brewing method like a coffee maker, French press, or pour-over."},
        {"input": "What is the largest planet in our solar system?", "output": "Jupiter is the largest planet in our solar system."}
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(filename, index=False)
    print(f"Sample dataset created: {filename}")

if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 4:
            print("Usage: python script.py test <model_path> <prompt>")
            sys.exit(1)
        test_model(sys.argv[2], sys.argv[3])
    
    # Check if creating sample dataset
    elif len(sys.argv) > 1 and sys.argv[1] == "create_sample":
        filename = sys.argv[2] if len(sys.argv) > 2 else "sample_dataset.csv"
        create_sample_dataset(filename)
    
    # Regular training
    else:
        main()
