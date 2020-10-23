export TASK_NAME=SST-3
export MODEL=xlnet-large-cased
export GLUE_DIR=/home/zhouchenchen/dataset/stocknews
export W_DIR=/home/zhouchenchen/pre-trained/$MODEL
export OUT_DIR=/home/zhouchenchen/myoutput/stocknews/$MODEL

CUDA_VISIBLE_DEVICES=1 nohup python /home/zhouchenchen/transformers/examples/text-classification/three_classification.py > stocknews/$MODEL.txt\
    --model_name_or_path $W_DIR \
    --task_name $TASK_NAME \
    --do_eval \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --learning_rate 2e-5 \
    --save_steps 25000 \
    --num_train_epochs 5.0 \
    --seed 42 \
    --overwrite_output_dir \
    --output_dir $OUT_DIR/


rm -rf /home/zhouchenchen/run/runs
rm /home/zhouchenchen/dataset/stocknews/SST-3/cached_*