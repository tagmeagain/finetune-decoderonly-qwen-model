# Qwen Fine-tuning with Unsloth

This project provides a complete setup for fine-tuning the Qwen model using Unsloth, based on the official Unsloth Colab notebook. Unsloth provides optimized fine-tuning with significant speed and memory improvements.

## Features

- **Unsloth Framework**: Optimized fine-tuning with 2x speed and 60% memory reduction
- **Qwen 1.5 Support**: Latest Qwen 1.5 model with improved performance
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with LoRA adapters
- **Alpaca Dataset Format**: Standard instruction-following dataset format
- **Docker Support**: Complete containerized setup
- **4-bit Quantization**: Memory-efficient training
- **Wandb Integration**: Training monitoring and logging
- **Multi-GPU Support**: Distributed training capabilities

## Project Structure

```
qwen-unsloth/
├── fine_tune_qwen.py        # Main fine-tuning script
├── inference.py             # Inference script
├── train.sh                 # Training script
├── alpaca_dataset.json      # Sample dataset in Alpaca format
├── Dockerfile              # Docker container definition
└── README.md               # This file
```

## Quick Start

### 1. Build Docker Image
```bash
docker build -t qwen-unsloth .
```

### 2. Run Container
```bash
docker run -d --name qwen-unsloth-container --gpus all -p 8888:8888 qwen-unsloth
```

### 3. Get Inside Container
```bash
docker exec -it qwen-unsloth-container bash
```

### 4. Start Training
```bash
./train.sh
```

## Dataset Format

The project uses the Alpaca dataset format, which is a standard for instruction-following models:

```json
[
  {
    "instruction": "What is machine learning?",
    "input": "",
    "output": "Machine learning is a subset of artificial intelligence..."
  },
  {
    "instruction": "Write a Python function to calculate factorial",
    "input": "Calculate factorial of 5",
    "output": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n\nresult = factorial(5)  # 120"
  }
]
```

### Dataset Fields:
- **instruction**: The task or question to be performed
- **input**: Additional context or input (can be empty)
- **output**: The expected response or answer

## Configuration

### Model Configuration
The main configuration is in `fine_tune_qwen.py`:

```python
# Configuration
MODEL_NAME = "Qwen/Qwen1.5-1.8B"
DATASET_PATH = "./alpaca_dataset.json"
OUTPUT_DIR = "./outputs"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
```

### Key Parameters:
- **MODEL_NAME**: Qwen model variant (1.8B, 7B, 14B, 72B)
- **MAX_SEQ_LENGTH**: Maximum sequence length for training
- **BATCH_SIZE**: Batch size per GPU
- **GRADIENT_ACCUMULATION_STEPS**: Effective batch size = batch_size × gradient_accumulation_steps
- **NUM_EPOCHS**: Number of training epochs
- **LEARNING_RATE**: Training learning rate

## Training

### Single GPU Training
```bash
python fine_tune_qwen.py
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_qwen.py
```

### Using the Training Script
```bash
./train.sh
```

## Inference

After training, test your model:

```bash
python inference.py --model_path ./outputs
```

### Inference Options:
- `--model_path`: Path to the trained model
- `--max_new_tokens`: Maximum number of tokens to generate
- `--temperature`: Sampling temperature (0.0-1.0)

## Performance Optimization

### Unsloth Optimizations
- **2x Training Speed**: Optimized CUDA kernels
- **60% Memory Reduction**: Efficient memory management
- **4-bit Quantization**: Reduced memory footprint
- **Flash Attention**: Faster attention computation
- **Gradient Checkpointing**: Memory-efficient training

### Hardware Recommendations

1. **Small GPUs (8-12GB)**
   - Use 1.8B model
   - LoRA rank 8-16
   - Batch size 1-2

2. **Medium GPUs (16-24GB)**
   - Use 7B model
   - LoRA rank 16-32
   - Batch size 2-4

3. **Large GPUs (32GB+)**
   - Use 14B or 72B model
   - LoRA rank 32-64
   - Batch size 4-8

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

1. **Memory Optimization**
   - Use 4-bit quantization (enabled by default)
   - Enable gradient checkpointing
   - Use appropriate LoRA rank
   - Reduce batch size if needed

2. **Speed Optimization**
   - Use Flash Attention (enabled by default)
   - Use mixed precision training
   - Use multiple GPUs
   - Enable gradient accumulation

3. **Quality Optimization**
   - Use higher LoRA rank
   - Increase training epochs
   - Use better quality training data
   - Tune learning rate

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

## Comparison with Other Frameworks

| Feature | Unsloth | Standard HF | Axolotl |
|---------|---------|-------------|---------|
| Training Speed | 2x faster | 1x | 1.5x |
| Memory Usage | 60% less | 100% | 70% |
| Ease of Use | High | Medium | High |
| Customization | High | High | Medium |
| Multi-GPU | Yes | Yes | Yes |

## License

This project is for educational and research purposes. Please ensure you comply with the licenses of the models and datasets you use.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the optimized fine-tuning framework
- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co/) for the transformers library 