# Qwen Fine-tuning with PEFT and SFTConfig

This project provides a simple and clean setup for fine-tuning the Qwen model using PEFT (Parameter-Efficient Fine-Tuning) with SFTConfig, without using Unsloth. It uses standard sentence transformers and Hugging Face libraries.

## Features

- **PEFT Framework**: Parameter-efficient fine-tuning with LoRA
- **SFTConfig**: Supervised Fine-Tuning configuration for better control
- **No Unsloth**: Simple, standard approach using Hugging Face libraries
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **Alpaca Dataset Format**: Standard instruction-following dataset format
- **Docker Support**: Complete containerized setup
- **4-bit Quantization**: Memory-efficient training
- **Wandb Integration**: Training monitoring and logging
- **Sentence Transformers**: Included for additional capabilities

## Project Structure

```
peft-code/
├── fine_tune_qwen_peft.py    # Main fine-tuning script with SFTConfig
├── inference.py              # Inference script
├── dataset.json              # Sample dataset with input/output format
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
└── README.md                # This file
```

## Quick Start

### 1. Build Docker Image
```bash
docker build -t qwen-peft .
```

### 2. Run Container
```bash
docker run -d --name qwen-peft-container --gpus all -p 8888:8888 qwen-peft
```

### 3. Get Inside Container
```bash
docker exec -it qwen-peft-container bash
```

### 4. Start Training
```bash
# Single GPU
python fine_tune_qwen_peft.py

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_qwen_peft.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 --master_port=29500 fine_tune_qwen_peft.py
```

## Configuration

### Model Configuration
The main configuration is in `fine_tune_qwen_peft.py`:

```python
# Configuration
MODEL_NAME = "Qwen/Qwen1.5-1.8B"
DATASET_PATH = "./dataset.json"
OUTPUT_DIR = "./outputs"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
```

### SFTConfig
```python
sft_config = SFTConfig(
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

## Dataset Format

The project uses a simple input/output dataset format:

```json
[
  {
    "input": "What is machine learning?",
    "output": "Machine learning is a subset of artificial intelligence..."
  }
]
```

### Dataset Fields:
- **input**: The question or input text
- **output**: The expected response or answer

## Training

### Single GPU Training
```bash
python fine_tune_qwen_peft.py
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_qwen_peft.py
```

### Using torchrun for Multi-GPU Training
```bash
# Single GPU
python fine_tune_qwen_peft.py

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_qwen_peft.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 --master_port=29500 fine_tune_qwen_peft.py

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 --master_port=29500 fine_tune_qwen_peft.py
```

## Inference

After training, test your model:

```bash
python inference.py --model_path ./outputs
```

### Inference Options:
- `--model_path`: Path to the trained model
- `--lora_path`: Path to LoRA adapter (optional)
- `--max_new_tokens`: Maximum number of tokens to generate
- `--temperature`: Sampling temperature (0.0-1.0)

## Key Differences from Unsloth

| Feature | PEFT (This Setup) | Unsloth |
|---------|-------------------|---------|
| Speed | Standard | 2x faster |
| Memory Usage | Standard | 60% less |
| Ease of Use | High | High |
| Customization | High | High |
| Dependencies | Standard HF | Custom |
| Installation | Simple | Complex |

## Performance Optimization

### Memory Optimization
- Use 4-bit quantization (enabled by default)
- Enable gradient checkpointing
- Use appropriate LoRA rank for your GPU
- Reduce batch size if needed

### Speed Optimization
- Use mixed precision training (bf16/fp16)
- Use multiple GPUs for distributed training
- Enable gradient accumulation
- Use appropriate batch size

### Quality Optimization
- Use higher LoRA rank for better quality
- Increase training epochs
- Use better quality training data
- Tune learning rate and scheduler

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space

### Recommended Requirements
- **GPU**: 24GB+ VRAM (for larger models)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space
- **Multiple GPUs**: For distributed training

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `BATCH_SIZE`
   - Increase `GRADIENT_ACCUMULATION_STEPS`
   - Use lower LoRA rank
   - Enable gradient checkpointing

2. **Slow Training**
   - Ensure GPU is being used
   - Check CUDA installation
   - Use mixed precision training
   - Consider using multiple GPUs

3. **Dataset Issues**
   - Verify JSON format matches Alpaca format
   - Check file encoding (UTF-8)
   - Ensure all required fields are present

4. **Model Loading Issues**
   - Check model path
   - Ensure sufficient disk space
   - Verify internet connection for model download

### Performance Tips

1. **For Small GPUs (8-12GB)**
   - Use 1.8B model
   - LoRA rank 8-16
   - Batch size 1-2

2. **For Medium GPUs (16-24GB)**
   - Use 7B model
   - LoRA rank 16-32
   - Batch size 2-4

3. **For Large GPUs (32GB+)**
   - Use 14B or 72B model
   - LoRA rank 32-64
   - Batch size 4-8

## Monitoring Training

### Wandb Integration
Training progress is automatically logged to Wandb if `WANDB_API_KEY` is set:
- View training metrics in real-time
- Compare different runs
- Monitor resource usage
- Export training logs

### Local Logging
Training logs are saved to `./logs/` directory:
- Training metrics
- Model checkpoints
- Evaluation results

## Advanced Configuration

### Custom LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### Custom SFT Configuration
```python
sft_config = SFTConfig(
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,
)
```

### Custom Training Arguments
```python
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    warmup_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="wandb",
)
```

## License

This project is for educational and research purposes. Please ensure you comply with the licenses of the models and datasets you use.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- [TRL](https://github.com/huggingface/trl) for SFTTrainer and SFTConfig
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co/) for the transformers library 