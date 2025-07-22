#!/bin/bash

# Example script to train Informer on crypto data
# Replace 'your_crypto_data.csv' with your actual CSV file path

echo "Training Informer on Crypto Data"

# Multivariate forecasting (using all features to predict all features)
echo "=== Multivariate Forecasting (M) ==="
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features M \
    --seq_len 168 \
    --label_len 48 \
    --pred_len 24 \
    --train_epochs 15 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --do_predict

echo ""
echo "=== Multivariate to Univariate Forecasting (MS) ==="
# Multivariate to univariate (use all features to predict only Close price)
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features MS \
    --target Close \
    --seq_len 168 \
    --label_len 48 \
    --pred_len 24 \
    --train_epochs 15 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --do_predict

echo ""
echo "=== Univariate Forecasting (S) ==="
# Univariate (use only Close price to predict Close price)
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features S \
    --target Close \
    --seq_len 168 \
    --label_len 48 \
    --pred_len 24 \
    --train_epochs 15 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --do_predict

echo "Training completed! Check the results folder for outputs."