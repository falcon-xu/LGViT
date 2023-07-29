path='/xxx/path/to/Early_Exit'
model_path='/xxx/path/to/Early_Exit/models/deit_highway'
export PYTHONPATH=$path:$PYTHONPATH         # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path

BACKBONE=DeiT
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=facebook/deit-base-distilled-patch16-224
DATASET=cifar100

export CUDA_VISIBLE_DEVICES=5,6,7
export WANDB_PROJECT=${BACKBONE}_${DATANAME}

python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 ../examples/run_base_deit.py \
    --report_to wandb \
    --run_name ${BACKBONE}-base \
    --dataset_name $DATASET \
    --backbone $BACKBONE \
    --model_name_or_path $MODEL_NAME \
    --output_dir ../saved_models/$MODEL_TYPE/$DATASET/base \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 777 \
    --ignore_mismatched_sizes=True
