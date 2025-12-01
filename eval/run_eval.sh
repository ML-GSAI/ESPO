#!/bin/bash

# Configuration variables

MASTER_PORT=33322

# Arrays of tasks and generation lengths
# TASKS=("sudoku")
TASKS=("countdown")

GEN_LENGTHS=(128 256 512)

NUM_GPUS=1
# Base model path
MODEL_PATH=/home/ubuntu/jianwen-us-south-2/jiaqi/assets/models/LLaDA-8B-Instruct
# MODEL_PATH="/home/ubuntu/jianwen-us-south-2/jiaqi/assets/models/Dream-v0-Instruct-7B"

# Lora model path
checkpoint_path="/home/ubuntu/jianwen-us-south-2/jiaqi/output/ablation/countdown_base_bs8_gspo_elbo_k2_mc2/checkpoints/checkpoint-9500"
# checkpoint_path="/home/ubuntu/jianwen-us-south-2/jiaqi/output/ablation/countdown_dream_bs8_grpo_meanfield_k3_mc2/checkpoints/checkpoint-7000"

output_dir="eval/countdown_base_bs8_gspo_elbo_k2_mc2/checkpoint-9500"
for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    batch_size=1
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir $output_dir \
      --model_path $MODEL_PATH \
      --checkpoint_path $checkpoint_path 

  done
done

python parse_and_get_acc.py --directory $output_dir >>$output_dir/summary.txt

echo "All evaluations completed!"
