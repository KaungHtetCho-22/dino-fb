#!/bin/bash

# Variables
DATA_PATH="/home/aic/leukemia/datasets/leukemia_dataset/train/"
OUTPUT_DIR="./dino_outputs_leukemia_patch4"
ARCH="vit_small"
PATCH_SIZE=8
EPOCHS=100
BATCH_SIZE=4
NUM_WORKERS=4
SAVE_FREQ=20

python main_dino.py \
  --arch $ARCH \
  --patch_size $PATCH_SIZE \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --epochs $EPOCHS \
  --batch_size_per_gpu $BATCH_SIZE \
  --use_fp16 true \
  --saveckp_freq $SAVE_FREQ \
  --num_workers $NUM_WORKERS
