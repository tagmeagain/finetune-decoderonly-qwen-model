#!/usr/bin/env python3
"""
Enhanced Fine-tuning script for Qwen 1.5B using Unsloth with LoRA
This script provides multiple LoRA configurations and better customization.
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
import wandb
from sklearn.model_selection import train_test_split
from lora_config import get_lora_config, apply_lora_config, print_available_configs

def load_and_prepare_dataset(csv_path, test_size=0.1, random_state=42):
    """
    Load CSV dataset and prepare it for training.
    
    Args:
        csv_path (str): Path to the CSV file
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    print(f"Loading dataset from {csv_path}")
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Clean the data
    df = df.dropna()  # Remove rows with missing values
    
    # Create conversation format
    conversations = []
    for _, row in df.iterrows():
        system_msg = row['System'] if pd.notna(row['System']) and row['System'].strip() else ""
        user_msg = row['user'].strip()
        assistant_msg = row['assistant'].strip()
        
        # Format conversation
        if system_msg:
            conversation = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        else:
            conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        
        conversations.append({"text": conversation})
    
    # Split into train and eval
    train_data, eval_data = train_test_split(
        conversations, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples")
    
    return train_dataset, eval_dataset

def setup_model_and_tokenizer(model_name="Qwen/Qwen1.5-1.8B", max_seq_length=2048, lora_config_name="balanced"):
    """
    Setup the model and tokenizer using Unsloth with LoRA.
    
    Args:
        model_name (str): Name of the model to load
        max_seq_length (int): Maximum sequence length
        lora_config_name (str): Name of the LoRA configuration to use
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # None for auto detection
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    # Get LoRA configuration
    lora_config = get_lora_config(lora_config_name)
    
    # Apply LoRA configuration
    model = apply_lora_config(model, lora_config)
    
    return model, tokenizer, lora_config

def create_training_arguments(output_dir="./qwen-finetuned", 
                            num_train_epochs=3,
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            learning_rate=2e-4,
                            warmup_steps=100,
                            logging_steps=10,
                            save_steps=500,
                            eval_steps=500):
    """
    Create training arguments.
    
    Returns:
        TrainingArguments: Training configuration
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if wandb.run else None,
        run_name="qwen-1.5b-lora-finetune",
    )

def tokenize_function(examples, tokenizer, max_length=2048):
    """
    Tokenize the examples for training.
    
    Args:
        examples: Dataset examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        dict: Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def print_training_info(model, lora_config, dataset_size, num_epochs, batch_size):
    """Print training information and statistics."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: Qwen 1.5B with LoRA")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameter Efficiency: {trainable_params/total_params*100:.2f}%")
    
    # LoRA info
    print(f"\nLoRA Configuration: {lora_config.name}")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target Modules: {len(lora_config.target_modules)} layers")
    print(f"  RSLoRA: {lora_config.use_rslora}")
    
    # Training info
    print(f"\nTraining Setup:")
    print(f"  Dataset Size: {dataset_size} samples")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lora_config.learning_rate}")
    
    # Memory estimation
    estimated_memory = (trainable_params * 4) / (1024**3)  # Rough estimate in GB
    print(f"  Estimated Memory: ~{estimated_memory:.2f} GB")
    
    print("="*60)

def main():
    """Main training function."""
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"
    DATASET_PATH = "sample_dataset.csv"
    OUTPUT_DIR = "./qwen-finetuned-lora"
    MAX_SEQ_LENGTH = 2048
    NUM_EPOCHS = 3
    BATCH_SIZE = 2
    
    # LoRA Configuration - Choose from: "minimal", "efficient", "balanced", "performance", "custom"
    LORA_CONFIG_NAME = "balanced"
    
    print("🚀 Qwen 1.5B LoRA Fine-tuning with Unsloth")
    print("=" * 50)
    
    # Show available LoRA configurations
    print("\nAvailable LoRA configurations:")
    print_available_configs()
    
    print(f"\nUsing LoRA configuration: {LORA_CONFIG_NAME}")
    
    # Initialize wandb (optional)
    try:
        wandb.init(
            project="qwen-lora-finetune", 
            name=f"qwen-1.5b-{LORA_CONFIG_NAME}",
            config={
                "model": MODEL_NAME,
                "lora_config": LORA_CONFIG_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_seq_length": MAX_SEQ_LENGTH
            }
        )
    except:
        print("Wandb not available, continuing without logging")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("⚠️  Warning: Training on CPU will be very slow. Consider using a GPU.")
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(DATASET_PATH)
    
    # Setup model and tokenizer with LoRA
    model, tokenizer, lora_config = setup_model_and_tokenizer(
        MODEL_NAME, 
        MAX_SEQ_LENGTH, 
        LORA_CONFIG_NAME
    )
    
    # Print training information
    print_training_info(model, lora_config, len(train_dataset), NUM_EPOCHS, BATCH_SIZE)
    
    # Tokenize datasets
    def tokenize_train(examples):
        return tokenize_function(examples, tokenizer, MAX_SEQ_LENGTH)
    
    def tokenize_eval(examples):
        return tokenize_function(examples, tokenizer, MAX_SEQ_LENGTH)
    
    train_dataset = train_dataset.map(tokenize_train, batched=True)
    eval_dataset = eval_dataset.map(tokenize_eval, batched=True)
    
    # Create training arguments with LoRA-specific learning rate
    training_args = create_training_arguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=lora_config.learning_rate
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n🚀 Starting LoRA fine-tuning...")
    print("This may take several hours. Monitor progress below.")
    print("-" * 50)
    
    trainer.train()
    
    # Save the final model
    print(f"\n💾 Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save LoRA configuration info
    config_info = {
        "lora_config_name": LORA_CONFIG_NAME,
        "lora_config": {
            "r": lora_config.r,
            "alpha": lora_config.lora_alpha,
            "dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
            "use_rslora": lora_config.use_rslora
        },
        "training_config": {
            "model": MODEL_NAME,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": lora_config.learning_rate,
            "max_seq_length": MAX_SEQ_LENGTH
        }
    }
    
    import json
    with open(os.path.join(OUTPUT_DIR, "lora_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    # Evaluate the model
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    print("\n🎉 LoRA fine-tuning completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"LoRA configuration: {lora_config.name}")
    
    # Clean up wandb
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main() 