#!/bin/bash

MODEL_PATHS=("google/mt5-small" "google/mt5-large" "cjvt/t5-sl-small" "cjvt/t5-sl-large")
MODEL_SHORTHANDS=("mt5-small" "mt5-large" "t5-sl-small" "t5-sl-large")
TRAIN_FILE="summarization/STA/train-concat.json" #test-concat.jsonl #~/summarization$ ls STA/
VALIDATION_FILE="summarization/STA/val-concat.json"
TEST_FILE="summarization/STA/test-concat.json"
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=512
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=2
NUM_TRAIN_EPOCHS=15
GRADIENT_ACCUMULATION=32


for i in 0 1 2 3
do
OUTPUT_DIR="t5-finetuning/hf-summar-sta-${MODEL_SHORTHANDS[$i]}-best-ckpt-only"
MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
python -m torch.distributed.launch --nproc_per_node 2 run_summarization.py\
 --model_name_or_path $MODEL_NAME_OR_PATH \
 --no_use_fast_tokenizer \
 --train_file $TRAIN_FILE \
 --validation_file $VALIDATION_FILE \
 --test_file $TEST_FILE \
 --text_column "article" \
 --summary_column "abstract" \
 --max_source_length $MAX_SOURCE_LENGTH \
 --max_target_length $MAX_TARGET_LENGTH \
 --output_dir $OUTPUT_DIR \
 --do_train --do_eval --do_predict \
 --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
 --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
 --num_train_epochs $NUM_TRAIN_EPOCHS \
 --save_strategy epoch \
 --evaluation_strategy epoch \
 --seed 42 \
 --generation_max_length $MAX_TARGET_LENGTH \
 --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
 --predict_with_generate\
 --load_best_model_at_end \
 --metric_for_best_model "eval_rougeL" \
 --greater_is_better=True \
 --save_total_limit=1


done
