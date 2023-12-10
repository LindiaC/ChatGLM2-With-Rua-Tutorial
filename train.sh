PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=2 # Set it to 2 because we have 2 GPUs.

# The parameters we should edit:
# --train_file # Change to your relative path of train.json
# --validation_file # Change to your relative path of dev.json
# --output_dir /kaggle/working/output/{Whatever} # to make sure you can download the output
# --max_steps n # As an example, a train.json with 180k lines needs 15s to train a step
# --save_steps n # you want to save a checkpoint every n step
# --quantization_bit 4

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file WechatMsg/train.json \
    --validation_file WechatMsg/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b-int4 \
    --output_dir /kaggle/working/output/ruabot-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4