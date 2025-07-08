#!/usr/bin/env python3
"""
Fine-tune Qwen model using PEFT with SFTConfig
Simple approach without Unsloth, using sentence transformers
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    SFTConfig
)
from trl import SFTTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import json
import os

def load_simple_dataset(file_path):
    """Load dataset with input and output columns"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format expected by SFTTrainer
    formatted_data = []
    for item in data:
        formatted_data.append({
            "input": item.get("input", ""),
            "output": item.get("output", "")
        })
    
    return Dataset.from_list(formatted_data)

def create_simple_prompt_format(input_text, output):
    """Create simple prompt format with EOS token"""
    prompt = f"Input: {input_text}\nOutput: {output}"
    return prompt

def formatting_prompts_func(examples, tokenizer):
    """Format prompts with EOS token - returns dict with 'text' field"""
    texts = []
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
    for input_text, output in zip(examples["input"], examples["output"]):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = create_simple_prompt_format(input_text, output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"
    DATASET_PATH = "./dataset.json"  # Changed to dataset.json
    OUTPUT_DIR = "./outputs"
    MAX_SEQ_LENGTH = 2048
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    
    print("🚀 Starting Qwen fine-tuning with PEFT and SFTConfig...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected, using CPU (this will be very slow)")
    
    # Load dataset
    print(f"📊 Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("Please create dataset.json with your input/output data")
        print("Expected format:")
        print('[\n  {"input": "your input text", "output": "your output text"}\n]')
        return
    
    dataset = load_simple_dataset(DATASET_PATH)
    print(f"✅ Loaded {len(dataset)} training examples")
    
    # Load model and tokenizer
    print(f"🤖 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("📝 Preparing dataset...")
    def formatting_func(examples):
        return formatting_prompts_func(examples, tokenizer)
    
    # Apply formatting to dataset
    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)
    print(f"✅ Dataset formatted with {len(dataset)} examples")
    
    # Configure SFT
    print("⚙️  Configuring SFT...")
    sft_config = SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        warmup_steps=WARMUP_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
    )
    
    # Create trainer
    print("🏋️  Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        sft_config=sft_config,
    )
    
    # Start training
    print("🎯 Starting training...")
    trainer_stats = trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    
    # Print training stats
    print("📈 Training completed!")
    print(f"   Total training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"   Training loss: {trainer_stats.training_loss:.4f}")
    print(f"   Model saved to: {OUTPUT_DIR}")
    
    # Test the model
    print("\n🧪 Testing the model...")
    
    # Sample test
    test_prompt = create_simple_prompt_format("What is machine learning?", "")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Sample response: {response}")

if __name__ == "__main__":
    main() 