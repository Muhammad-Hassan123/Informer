# Crypto Price Prediction with Informer

This guide shows you how to train the Informer model on your cryptocurrency data for price forecasting.

## 📋 Data Requirements

Your CSV file should have the following columns (in any order):
- `Open Time` - Timestamp (milliseconds or seconds)
- `Open` - Opening price
- `High` - Highest price in the period
- `Low` - Lowest price in the period
- `Close` - Closing price (target for prediction)
- `Volume` - Trading volume
- `Close Time` - Close timestamp
- `Quote Asset Volume` - Quote asset volume
- `Number of Trades` - Number of trades
- `Taker Buy Base Volume` - Taker buy base volume
- `Taker Buy Quote Volume` - Taker buy quote volume

## 🚀 Quick Start

### 1. Prepare Your Data
Place your crypto CSV file in the project directory. The script will automatically:
- Convert timestamps to datetime
- Select relevant features
- Sort data chronologically
- Save processed data to `./data/` folder

### 2. Basic Training Command

```bash
python crypto_train.py --crypto_data your_crypto_file.csv --coin_name BTC
```

### 3. Advanced Training Examples

#### Multivariate Forecasting (All features → All features)
```bash
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features M \
    --seq_len 168 \
    --pred_len 24 \
    --train_epochs 20 \
    --do_predict
```

#### Price Prediction Only (All features → Close price)
```bash
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features MS \
    --target Close \
    --seq_len 168 \
    --pred_len 24 \
    --train_epochs 20 \
    --do_predict
```

#### Univariate Price Prediction (Close → Close)
```bash
python crypto_train.py \
    --crypto_data your_crypto_data.csv \
    --coin_name BTC \
    --features S \
    --target Close \
    --seq_len 168 \
    --pred_len 24 \
    --train_epochs 20 \
    --do_predict
```

## 🔧 Key Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--seq_len` | Input sequence length (hours/days of history) | 96 (4 days), 168 (7 days), 336 (14 days) |
| `--pred_len` | Prediction horizon (hours/days to forecast) | 24 (1 day), 48 (2 days), 168 (7 days) |
| `--features` | Forecasting type | `M` (multivariate), `MS` (multi→uni), `S` (univariate) |
| `--train_epochs` | Number of training epochs | 10-30 |
| `--batch_size` | Training batch size | 16, 32, 64 |
| `--learning_rate` | Learning rate | 0.0001, 0.0005 |

## 📊 Understanding Output

After training, you'll find:

1. **Model Checkpoints**: Saved in `./checkpoints/`
2. **Training Logs**: Console output with loss metrics
3. **Predictions**: If `--do_predict` is used, predictions are saved
4. **Test Results**: Model evaluation on test set

### Metrics Explained
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## 💡 Tips for Better Results

### 1. Sequence Length Selection
- **Short sequences (24-96)**: Good for short-term predictions
- **Long sequences (168-336)**: Better for capturing weekly/monthly patterns
- **Very long sequences (720+)**: For long-term trends (requires more data)

### 2. Feature Selection
- **Multivariate (M)**: Use when you want to predict all features
- **Multivariate-to-Univariate (MS)**: Best for price prediction using all available data
- **Univariate (S)**: Fastest training, good baseline

### 3. Training Tips
- Start with fewer epochs (10-15) to test setup
- Increase batch size if you have GPU memory
- Use early stopping (patience=3) to avoid overfitting
- Monitor validation loss during training

### 4. Data Quality
- Ensure your data has no missing values
- More data (800+ days) generally gives better results
- Higher frequency data (hourly vs daily) can capture more patterns

## 🎯 Example Workflow

```bash
# 1. Quick test with small parameters
python crypto_train.py \
    --crypto_data btc_data.csv \
    --coin_name BTC \
    --seq_len 96 \
    --pred_len 24 \
    --train_epochs 5

# 2. If it works, scale up
python crypto_train.py \
    --crypto_data btc_data.csv \
    --coin_name BTC \
    --features MS \
    --seq_len 168 \
    --pred_len 24 \
    --train_epochs 20 \
    --batch_size 32 \
    --do_predict

# 3. Try different prediction horizons
python crypto_train.py \
    --crypto_data btc_data.csv \
    --coin_name BTC \
    --features MS \
    --seq_len 336 \
    --pred_len 168 \
    --train_epochs 25 \
    --do_predict
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `batch_size` (try 8 or 16)
   - Reduce `seq_len` or `d_model`

2. **Poor Performance**
   - Increase `train_epochs`
   - Try different `seq_len` values
   - Check data quality and preprocessing

3. **Training Too Slow**
   - Increase `batch_size` if you have GPU memory
   - Reduce `seq_len` for faster training
   - Use fewer `train_epochs` initially

4. **File Not Found Errors**
   - Ensure your CSV file path is correct
   - Check that `./data/` directory exists

### GPU Usage
- The script automatically detects and uses GPU if available
- For CPU-only training, add `--use_gpu False`

## 📈 Expected Results

With 800 days of crypto data, you can expect:
- **Training time**: 10-60 minutes depending on parameters
- **Prediction accuracy**: Varies by market conditions and parameters
- **Best results**: Usually with `features=MS`, `seq_len=168`, `pred_len=24`

The model will learn patterns like:
- Daily/weekly price cycles
- Volume-price relationships  
- Technical indicator patterns
- Long-term trends

Remember: Crypto markets are highly volatile and unpredictable. Use predictions as one factor among many in your analysis!