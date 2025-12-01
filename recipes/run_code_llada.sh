export E2B_API_KEY=key
DATASET="code"
export LOGDIR=output
NUM_ITER=8
RUN_NAME=${DATASET}_base_mu_${NUM_ITER}_gspo_llada_mc_2
model_name_or_path='GSAI-ML/LLaDA-8B-Instruct'
# set wandb output path
export WANDB_DIR="$LOGDIR/$RUN_NAME"

# create output directory
mkdir -p "$LOGDIR/$RUN_NAME"

# ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 8 \
    espo/espo_train.py --config recipes/train.yaml \
    --dataset_name "$DATASET" \
    --model_name_or_path ${model_name_or_path} \
    --run_name "$RUN_NAME" \
    --logging_steps 1 \
    --num_iterations "$NUM_ITER" \
    --gradient_accumulation_steps 20 \
    --per_device_train_batch_size 1 \
    --num_generations 10 \
    --generation_batch_size 10 \
    --num_mc 2 \
    --max_grad_norm 0.8 \
    --warmup_ratio 0.001 \
    --max_prompt_length 400 \
    --beta 1e-2 \
    --code_provider "e2b" \
    --lr_scheduler_type constant_with_warmup \
    --output_dir "$LOGDIR/$RUN_NAME/checkpoints" \
    --use_peft false \
    --gradient_checkpointing true \
    --diffusion_steps 256 \
    --wandb_project "espo"  \