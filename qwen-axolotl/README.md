# Qwen 2.1.5 Fine-tuning with Axolotl

This project provides a complete setup for fine-tuning the Qwen 2.1.5 model using Axolotl, a powerful and efficient fine-tuning framework.

## Features

- **Axolotl Framework**: Uses Axolotl for efficient and optimized fine-tuning
- **Qwen 2.1.5 Support**: Latest Qwen 2.1.5 model with improved performance
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with LoRA adapters
- **Flash Attention**: Optimized attention mechanism for faster training
- **Docker Support**: Complete containerized setup
- **CSV Dataset Support**: Easy data loading from CSV files
- **Wandb Integration**: Training monitoring and logging
- **Multi-GPU Support**: Distributed training capabilities

## Project Structure

```
qwen-axolotl/
├── train.yml                 # Axolotl configuration file
├── train.sh                  # Training script
├── inference.py              # Inference script
├── sample_dataset.csv        # Sample dataset
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
└── README.md                # This file
```

## Quick Start

### 1. Build Docker Image
```bash
docker build -t qwen-axolotl .
```

### 2. Run Container
```bash
docker run -d --name qwen-axolotl-container --gpus all -p 8888:8888 qwen-axolotl
```

### 3. Get Inside Container
```bash
docker exec -it qwen-axolotl-container bash
```

### 4. Start Training
```bash
./train.sh
```

## Configuration

### Model Configuration (`train.yml`)

The main configuration file `train.yml` contains all training parameters:

```yaml
# Model configuration
base_model: Qwen/Qwen2.1.5-1.8B
model_type: Qwen2ForCausalLM
trust_remote_code: true

# LoRA configuration
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# Training configuration
num_epochs: 3
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
```

### Key Parameters

- **base_model**: Qwen 2.1.5 model variant (1.8B, 7B, 14B, 72B)
- **lora_r**: LoRA rank (higher = more parameters, better quality)
- **micro_batch_size**: Batch size per GPU
- **gradient_accumulation_steps**: Effective batch size = micro_batch_size × gradient_accumulation_steps
- **learning_rate**: Training learning rate

## Dataset Format

Your CSV dataset should have the following columns:
- `System`: System prompts (can be empty)
- `user`: User questions/queries
- `assistant`: Assistant responses/answers

Example:
```csv
System,user,assistant
,"What is machine learning?","Machine learning is a subset of artificial intelligence..."
,"How does a neural network work?","A neural network is a computational model..."
```

## Training

### Single GPU Training
```bash
axolotl train train.yml
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 --master_port=29500 -m axolotl.cli.train train.yml
```

### Custom Configuration
```bash
# Override configuration parameters
axolotl train train.yml --num_epochs 5 --learning_rate 1e-4
```

## Inference

After training, test your model:

```bash
python inference.py --model_path ./outputs --lora_path ./lora_outputs
```

### Inference Options
- `--model_path`: Path to the trained model
- `--lora_path`: Path to LoRA adapter (optional)
- `--max_length`: Maximum sequence length
- `--temperature`: Sampling temperature

## Performance Optimization

### Memory Optimization
- Use 4-bit quantization (enabled by default)
- Enable gradient checkpointing
- Use appropriate LoRA rank for your GPU
- Reduce batch size if needed

### Speed Optimization
- Use Flash Attention (enabled by default)
- Use mixed precision training (bf16)
- Use multiple GPUs for distributed training
- Enable gradient accumulation

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
   - Reduce `micro_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use lower LoRA rank
   - Enable gradient checkpointing

2. **Slow Training**
   - Ensure Flash Attention is enabled
   - Check GPU utilization
   - Use mixed precision training
   - Consider using multiple GPUs

3. **Model Loading Issues**
   - Check model path in configuration
   - Ensure sufficient disk space
   - Verify internet connection for model download

4. **Dataset Issues**
   - Verify CSV format matches expected columns
   - Check file permissions
   - Ensure proper encoding (UTF-8)

### Performance Tips

1. **For Small GPUs (8-12GB)**
   - Use 1.8B model
   - LoRA rank 8-16
   - Micro batch size 1-2

2. **For Medium GPUs (16-24GB)**
   - Use 7B model
   - LoRA rank 16-32
   - Micro batch size 2-4

3. **For Large GPUs (32GB+)**
   - Use 14B or 72B model
   - LoRA rank 32-64
   - Micro batch size 4-8

## Monitoring Training

### Wandb Integration
Training progress is automatically logged to Wandb. You can:
- View training metrics in real-time
- Compare different runs
- Monitor resource usage
- Export training logs

### Local Logging
Training logs are saved to `./logs/` directory:
- Tensorboard logs
- Training metrics
- Model checkpoints

## Advanced Configuration

### Custom LoRA Configuration
```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### DeepSpeed Integration
```yaml
deepspeed: deepspeed_config.json
```

### Custom Training Arguments
```yaml
warmup_steps: 100
logging_steps: 10
save_steps: 500
eval_steps: 500
save_total_limit: 3
```

## License

This project is for educational and research purposes. Please ensure you comply with the licenses of the models and datasets you use.

## Contributing

Feel free to submit issues and enhancement requests! 