#!/usr/bin/env python3
"""
Fine-tune Qwen model using Unsloth
Based on the Unsloth Colab notebook
"""

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import json
import os
from unsloth import FastLanguageModel

def load_alpaca_dataset(file_path):
    """Load dataset in Alpaca format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format expected by Unsloth
    formatted_data = []
    for item in data:
        formatted_data.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        })
    
    return Dataset.from_list(formatted_data)

def create_alpaca_prompt_format(instruction, input_text, output):
    """Create prompt in the Alpaca format with EOS token"""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    if input_text:
        prompt = alpaca_prompt.format(instruction, input_text, output)
    else:
        prompt = alpaca_prompt.format(instruction, "", output)
    
    return prompt

def formatting_prompts_func_with_eos(examples, tokenizer):
    """Format prompts with EOS token"""
    prompts = []
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        prompt = create_alpaca_prompt_format(instruction, input_text, output) + EOS_TOKEN
        prompts.append(prompt)
    return prompts

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen1.5-1.8B"
    DATASET_PATH = "./alpaca_dataset.json"
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
    
    print("🚀 Starting Qwen fine-tuning with Unsloth...")
    
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
        print("Please create alpaca_dataset.json with your data")
        return
    
    dataset = load_alpaca_dataset(DATASET_PATH)
    print(f"✅ Loaded {len(dataset)} training examples")
    
    # Load model and tokenizer
    print(f"🤖 Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # None for auto detection
        load_in_4bit=True,  # Use 4bit quantization
    )
    
    # Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Prepare dataset
    print("📝 Preparing dataset...")
    def formatting_prompts_func(examples):
        return formatting_prompts_func_with_eos(examples, tokenizer)
    
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
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        formatting_func=formatting_prompts_func,
        packing=False,
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
    FastLanguageModel.for_inference(model)
    
    # Sample test
    test_prompt = create_alpaca_prompt_format("What is machine learning?", "", "")
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    
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