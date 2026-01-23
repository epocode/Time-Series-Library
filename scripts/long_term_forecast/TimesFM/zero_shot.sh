#!/bin/bash

# TimesFM 零样本预测实验脚本
# 根据论文 "A decoder-only foundation model for time-series forecasting" (Das et al., 2024)
# 复现论文 Table 4 中的 ETT 数据集实验结果

# 通用参数
TASK_NAME="zero_shot_forecast"
IS_TRAINING=0
MODEL="TimesFM"
FEATURES="M"
SEQ_LEN=512
LABEL_LEN=48
ENC_IN=7
DEC_IN=7
C_OUT=7
DES="Exp"
ITR=1
USE_GPU=True
NUM_WORKERS=1
BATCH_SIZE=32
USE_AMP="--use_amp"

echo "=========================================="
echo "TimesFM Zero-Shot Forecasting Experiments"
echo "Reproduc ing results from Das et al., 2024"
echo "=========================================="
echo ""

# ==========================================
# ETTh1 数据集实验
# ==========================================
echo "Running ETTh1 experiments..."

# ETTh1, Horizon=96
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_96 \
  --model $MODEL \
  --data ETTh1 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 96 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ETTh1, Horizon=192
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_192 \
  --model $MODEL \
  --data ETTh1 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 192 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ==========================================
# ETTh2 数据集实验
# ==========================================
echo "Running ETTh2 experiments..."

# ETTh2, Horizon=96
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${SEQ_LEN}_96 \
  --model $MODEL \
  --data ETTh2 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 96 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ETTh2, Horizon=192
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_${SEQ_LEN}_192 \
  --model $MODEL \
  --data ETTh2 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 192 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ==========================================
# ETTm1 数据集实验
# ==========================================
echo "Running ETTm1 experiments..."

# ETTm1, Horizon=96
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${SEQ_LEN}_96 \
  --model $MODEL \
  --data ETTm1 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 96 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ETTm1, Horizon=192
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${SEQ_LEN}_192 \
  --model $MODEL \
  --data ETTm1 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 192 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ==========================================
# ETTm2 数据集实验
# ==========================================
echo "Running ETTm2 experiments..."

# ETTm2, Horizon=96
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${SEQ_LEN}_96 \
  --model $MODEL \
  --data ETTm2 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 96 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

# ETTm2, Horizon=192
python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_${SEQ_LEN}_192 \
  --model $MODEL \
  --data ETTm2 \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len 192 \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --des $DES \
  --itr $ITR \
  --use_gpu $USE_GPU \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  $USE_AMP

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: result_zero_shot_forecast_search.txt"
echo "=========================================="