# Docker Commands

## Build the Docker image
```bash
docker build -t qwen-finetune .
```

## Run the container
```bash
docker run -d --name qwen-container --gpus all -p 8888:8888 qwen-finetune
```

## Get inside the container
```bash
docker exec -it qwen-container bash
```

## Once inside the container, run distributed training
```bash
# For 2 GPUs
torchrun --nproc_per_node=2 --master_port=29500 fine_tune_lora.py

# For 4 GPUs  
torchrun --nproc_per_node=4 --master_port=29500 fine_tune_lora.py

# For 8 GPUs
torchrun --nproc_per_node=8 --master_port=29500 fine_tune_lora.py
```

## Stop and remove container
```bash
docker stop qwen-container
docker rm qwen-container
```

## View container logs
```bash
docker logs qwen-container
```

## Check GPU usage inside container
```bash
docker exec -it qwen-container nvidia-smi
``` 