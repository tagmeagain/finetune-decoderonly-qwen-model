# Docker Commands for PEFT Fine-tuning

## Quick Start Commands

### 1. Build the Docker Image
```bash
docker build -t qwen-peft .
```

### 2. Run Container with GPU Support
```bash
docker run -d --name qwen-peft-container --gpus all -p 8888:8888 qwen-peft
```

### 3. Access Container
```bash
docker exec -it qwen-peft-container bash
```

### 4. Start Training
```bash
./train.sh
```

## Alternative Commands

### Run Container with Volume Mounting
```bash
docker run -d --name qwen-peft-container \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  qwen-peft
```

### Run Container with Custom Dataset
```bash
docker run -d --name qwen-peft-container \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/your_dataset.json:/app/alpaca_dataset.json \
  -v $(pwd)/outputs:/app/outputs \
  qwen-peft
```

### Run Container with Environment Variables
```bash
docker run -d --name qwen-peft-container \
  --gpus all \
  -p 8888:8888 \
  -e WANDB_API_KEY=your_wandb_key \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  qwen-peft
```

## Management Commands

### Stop Container
```bash
docker stop qwen-peft-container
```

### Start Container
```bash
docker start qwen-peft-container
```

### Remove Container
```bash
docker rm qwen-peft-container
```

### View Container Logs
```bash
docker logs qwen-peft-container
```

### View Container Logs (Follow)
```bash
docker logs -f qwen-peft-container
```

### Check Container Status
```bash
docker ps -a
```

### Check GPU Usage Inside Container
```bash
docker exec -it qwen-peft-container nvidia-smi
```

## Multi-GPU Commands

### Run with Multiple GPUs
```bash
docker run -d --name qwen-peft-container \
  --gpus '"device=0,1"' \
  -p 8888:8888 \
  qwen-peft
```

### Run with Specific GPUs
```bash
docker run -d --name qwen-peft-container \
  --gpus '"device=0"' \
  -p 8888:8888 \
  qwen-peft
```

## Troubleshooting Commands

### Check Docker Version
```bash
docker --version
```

### Check NVIDIA Docker Support
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Check Available GPUs
```bash
nvidia-smi
```

### Clean Up Docker
```bash
docker system prune -a
```

### Rebuild Image (Force)
```bash
docker build --no-cache -t qwen-peft .
```

## Development Commands

### Run Container in Interactive Mode
```bash
docker run -it --rm --gpus all -p 8888:8888 qwen-peft bash
```

### Run Container with Jupyter
```bash
docker run -d --name qwen-peft-jupyter \
  --gpus all \
  -p 8888:8888 \
  -e JUPYTER_ENABLE_LAB=yes \
  qwen-peft jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Copy Files to Container
```bash
docker cp your_file.json qwen-peft-container:/app/
```

### Copy Files from Container
```bash
docker cp qwen-peft-container:/app/outputs ./local_outputs
```

## Performance Monitoring

### Monitor Container Resources
```bash
docker stats qwen-peft-container
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Memory Usage
```bash
docker exec -it qwen-peft-container free -h
```

### Check Disk Usage
```bash
docker exec -it qwen-peft-container df -h
``` 