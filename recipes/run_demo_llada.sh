DATASET="sudoku"
export LOGDIR=output
NUM_ITER=8
RUN_NAME=${DATASET}_base_mu_${NUM_ITER}_gspo_llada_mc2_demo
model_name_or_path='GSAI-ML/LLaDA-8B-Instruct'
# set wandb output path
export WANDB_DIR="$LOGDIR/$RUN_NAME"

# create output directory
mkdir -p "$LOGDIR/$RUN_NAME"

# ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 1 \
    espo/espo_train.py --config recipes/train.yaml \
    --dataset_name "$DATASET" \
    --model_name_or_path ${model_name_or_path} \
    --run_name "$RUN_NAME" \
    --logging_steps 1 \
    --num_iterations "$NUM_ITER" \
    --gradient_accumulation_steps 12 \
    --per_device_train_batch_size 1 \
    --generation_batch_size 6 \
    --num_generations 6 \
    --num_mc 1 \
    --max_grad_norm  0.2 \
    --warmup_ratio 0.001 \
    --max_prompt_length 400 \
    --beta 1e-3 \
    --epsilon 4e-4 \
    --learning_rate 3e-6 \
    --weight_decay 0.01 \
    --lr_scheduler_type constant_with_warmup \
    --output_dir "$LOGDIR/$RUN_NAME/checkpoints" \
    --use_peft true \
    --diffusion_steps 64  \
    --reduce_var false \
    --wandb_project "espo"  \
    # --gradient_checkpointing true  # use if out of memory


