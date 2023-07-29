path='/xxx/path/to/Early_Exit'
model_path_deit='/xxx/path/to/Early_Exit/models/deit_highway'
model_path='/xxx/path/to/Early_Exit/models/swin_highway'
export PYTHONPATH=$path:$PYTHONPATH         # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path_deit" # Add the model_path_deit to the end of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path

BACKBONE=Swin
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=microsoft/swin-base-patch4-window7-224
DATASET=cifar100      # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k

if [ $DATASET = 'Maysee/tiny-imagenet' ]; then
  DATANAME=tiny-imagenet
else
  DATANAME=$DATASET
fi

EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
PAPER_NAME=LGViT     # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT

export CUDA_VISIBLE_DEVICES=1,2,3
# export WANDB_PROJECT=${BACKBONE}_${DATANAME}eval

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29527 --nnodes=1 ../examples/run_highway_swin.py \
    --run_name Swin_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path /xxx/path/to/Early_Exit/checkpoint_path \
    --dataset_name $DATASET \
    --output_dir /home/haojiawei/Early_Exit/outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/ \
    --remove_unused_columns False \
    --exit_strategy $EXIT_STRATEGY \
    --num_early_exits [0,1,6,1] \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to wandb \
    --use_auth_token False \
