#!/usr/bin/env python3
"""
Qwen2 1.5B LoRA Fine-tuning Script
Supports CSV/JSON datasets with 'input' and 'output' columns
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset
import argparse
from typing import Dict, List

class QwenDataset(Dataset):
    """Custom dataset class for Qwen2 fine-tuning"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the conversation
        prompt = f"Human: {item['input']}\n\nAssistant: {item['output']}"
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

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

def main():
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
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} samples")
    
    # Validate dataset format
    if not all('input' in item and 'output' in item for item in data):
        raise ValueError("Dataset must contain 'input' and 'output' columns")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create dataset
    train_dataset = QwenDataset(data, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
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
    
    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

def test_model(model_path: str, prompt: str):
    """Test the fine-tuned model"""
    from peft import PeftModel
    
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response:\n{response}")

if __name__ == "__main__":
    # Check if running in test mode
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
        if len(os.sys.argv) < 4:
            print("Usage: python script.py test <model_path> <prompt>")
            exit(1)
        test_model(os.sys.argv[2], os.sys.argv[3])
    else:
        main()
