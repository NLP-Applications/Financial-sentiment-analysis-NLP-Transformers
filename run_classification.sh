export TASK_NAME=SST-3 %the task of three classification
export MODEL=[name of pre-trained model in huggingface]
export GLUE_DIR=[path to your dataset]
export W_DIR=[path to your model saved locally]
export OUT_DIR=[path to your output-dir]

CUDA_VISIBLE_DEVICES=1 nohup python three_classification.py > [path to your output log]\
    --model_name_or_path $MODE/$W_DIR \
    --task_name $TASK_NAME \
    --do_eval \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length [] \
    --per_device_eval_batch_size=[] \
    --per_device_train_batch_size=[] \
    --learning_rate [] \
    --save_steps [] \
    --num_train_epochs [] \
    --seed 42 \
    --overwrite_output_dir \
    --output_dir $OUT_DIR/
