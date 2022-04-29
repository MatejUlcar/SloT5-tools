#!/bin/bash

MODEL_PATHS=("google/mt5-small" "google/mt5-large" "cjvt/t5-sl-small" "cjvt/t5-sl-large")
MODEL_SHORTHANDS=("mt5-small" "mt5-large" "t5-sl-small" "t5-sl-large")
TRAIN_FILE="superglue/SuperGLUE-GoogleMT/json/COPA/train.jsonl.t5.json"
VALIDATION_FILE="superglue/SuperGLUE-GoogleMT/json/COPA/val.jsonl.t5.json"
TEST_FILE="superglue/superglue-test-sets/COPA/test_answered.jsonl.t5.json"
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=6
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
NUM_TRAIN_EPOCHS=15
GRADIENT_ACCUMULATION=8

for i in 0 1 2 3
do
OUTPUT_DIR="t5-finetuning/hf-copa-${MODEL_SHORTHANDS[$i]}-best-ckpt-only"
MODEL_NAME_OR_PATH=${MODEL_PATHS[$i]}
python run_summarization.py\
 --model_name_or_path $MODEL_NAME_OR_PATH \
 --no_use_fast_tokenizer \
 --train_file $TRAIN_FILE \
 --validation_file $VALIDATION_FILE \
 --test_file $TEST_FILE \
 --text_column "inputs" \
 --summary_column "targets" \
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
 --predict_with_generate \
 --load_best_model_at_end \
 --metric_for_best_model "eval_rougeL" \
 --greater_is_better=True \
 --save_total_limit=1
done
