path='/xxx/path/to/Early_Exit'
model_path_deit='/xxx/path/to/Early_Exit/models/deit_highway'
model_path='/xxx/path/to/Early_Exit/models/swin_highway'
export PYTHONPATH=$path:$PYTHONPATH         # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path_deit" # Add the model_path_deit to the end of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path

BACKBONE=Swin
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=microsoft/swin-base-patch4-window7-224
DATASET=cifar100       # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k
ADD_INFO=Swin_base

export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_PROJECT=${BACKBONE}_${DATANAME}

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29507 --nnodes=1 ../examples/run_base_swin.py \
    --report_to wandb \
    --run_name swin_base_${DATANAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET \
    --output_dir ../saved_models/$MODEL_TYPE/$DATASET/$ADD_INFO \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --learning_rate 5e-5 \
    --output_hidden_states False \
    --do_train \
    --do_eval \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 777 \
    --ignore_mismatched_sizes True \
    --use_auth_token False \
