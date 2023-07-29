path='/xxx/path/to/Early_Exit'
model_path='/xxx/path/to/Early_Exit/models/deit_highway'
export PYTHONPATH=$path:$PYTHONPATH         # Add path to the beginning of the search path
export PYTHONPATH="$PYTHONPATH:$model_path" # Add the model path to the end of the search path

BACKBONE=ViT # ViT, DeiT
MODEL_TYPE=${BACKBONE}-base
MODEL_NAME=facebook/deit-base-distilled-patch16-224
DATASET=cifar100      # cifar100, Food101, Maysee/tiny-imagenet, imagenet-1k

if [ $DATASET = 'Maysee/tiny-imagenet' ]; then
  DATANAME=tiny-imagenet
else
  DATANAME=$DATASET
fi

EXIT_STRATEGY=confidence # entropy, confidence, patience, patient_and_confident
HIGHWAY_TYPE=LGViT # linear, LGViT, vit, self_attention, conv_normal
PAPER_NAME=LGViT  # base, SDN, PABEE, PCEE, BERxiT, ViT-EE, LGViT

export CUDA_VISIBLE_DEVICES=0,1,2
#export WANDB_PROJECT=${BACKBONE}_${DATANAME}_eval

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29566 --nnodes=1 ../examples/run_highway.py \
    --run_name ${BACKBONE}_${EXIT_STRATEGY}_${HIGHWAY_TYPE}_${TRAIN_STRATEGY}_${PAPER_NAME} \
    --image_processor_name $MODEL_NAME \
    --config_name $MODEL_NAME \
    --model_name_or_path /xxx/path/to/Early_Exit/checkpoint_path \
    --dataset_name $DATASET \
    --output_dir ../outputs/$MODEL_TYPE/$DATASET/$PAPER_NAME/$EXIT_STRATEGY/ \
    --remove_unused_columns False \
    --backbone $BACKBONE \
    --exit_strategy $EXIT_STRATEGY \
    --do_train False \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --seed 777 \
    --report_to wandb \
    --use_auth_token False \
