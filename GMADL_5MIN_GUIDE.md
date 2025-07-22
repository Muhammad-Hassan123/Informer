# 🚀 GMADL Loss Function for 5-Minute Crypto Trading

## 🎯 **What is GMADL Loss?**

**GMADL (Generalized Mean Absolute Deviation Loss)** is a robust loss function that's perfect for crypto price prediction because it:

- ✅ **More robust to outliers** than MSE (handles crypto volatility better)
- ✅ **Better gradient flow** than pure MAE (faster convergence)
- ✅ **Tunable sensitivity** with the beta parameter
- ✅ **Optimized for time series** forecasting

### **Mathematical Formula:**
```
GMADL = (|predicted - actual|^β)
```

Where β controls the loss behavior:
- **β = 1.0**: Pure MAE (robust but slow convergence)
- **β = 2.0**: Pure MSE (fast convergence but sensitive to outliers)
- **β = 1.5**: Sweet spot for crypto (balanced robustness and speed)

## 🕐 **5-Minute Interval Advantages**

Training on 5-minute intervals gives you:
- ⚡ **High-frequency patterns**: Capture intraday trading patterns
- 📊 **More data points**: 288 intervals per day vs 24 hourly
- 🎯 **Short-term accuracy**: Better for day trading and scalping
- 📈 **Volatility capture**: Catch rapid price movements

## 🚀 **Quick Start Commands**

### **Basic GMADL Training (Recommended)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data your_5min_data.csv \
    --coin_name BTC \
    --loss gmadl \
    --beta 1.5 \
    --do_predict
```

### **Adaptive GMADL (Advanced)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data your_5min_data.csv \
    --coin_name BTC \
    --loss adaptive_gmadl \
    --beta_start 1.2 \
    --beta_end 1.8 \
    --train_epochs 20 \
    --do_predict
```

### **Weighted GMADL (Focus on Recent Predictions)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data your_5min_data.csv \
    --coin_name BTC \
    --loss weighted_gmadl \
    --beta 1.5 \
    --weight_decay_loss 0.95 \
    --do_predict
```

## 🔧 **Configuration Options**

### **Loss Function Types**

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `gmadl` | Standard GMADL with fixed beta | General crypto prediction |
| `adaptive_gmadl` | Beta changes during training | Long training runs |
| `weighted_gmadl` | More weight on recent predictions | Short-term trading |
| `mse` | Standard Mean Squared Error | Baseline comparison |
| `mae` | Mean Absolute Error | Very noisy data |

### **Beta Parameter Guide**

| Beta Value | Behavior | Use Case |
|------------|----------|----------|
| 1.0 | Pure MAE - very robust | Extremely volatile coins |
| 1.2 | Robust with some speed | High volatility |
| 1.5 | **Recommended** balance | Most crypto pairs |
| 1.8 | Fast convergence | Stable coins |
| 2.0 | Pure MSE - fastest | Low volatility assets |

### **5-Minute Interval Settings**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `seq_len` | 288 | 24 hours of 5-min data (288 × 5min) |
| `label_len` | 144 | 12 hours start token (144 × 5min) |
| `pred_len` | 72 | 6 hours prediction (72 × 5min) |
| `freq` | '5min' | Time feature encoding |
| `batch_size` | 16 | Smaller for high-frequency data |

## 🎯 **Recommended Workflows**

### **1. Quick Test (5 minutes)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data data/btc_5min.csv \
    --coin_name BTC \
    --seq_len 144 \
    --pred_len 36 \
    --train_epochs 3
```

### **2. Standard Training (30 minutes)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data data/btc_5min.csv \
    --coin_name BTC \
    --loss gmadl \
    --beta 1.5 \
    --seq_len 288 \
    --pred_len 72 \
    --train_epochs 15 \
    --do_predict
```

### **3. Production Training (1-2 hours)**
```bash
python crypto_train_gmadl_5min.py \
    --crypto_data data/btc_5min.csv \
    --coin_name BTC \
    --loss adaptive_gmadl \
    --beta_start 1.2 \
    --beta_end 1.8 \
    --seq_len 576 \
    --pred_len 144 \
    --train_epochs 25 \
    --batch_size 8 \
    --do_predict
```

## 📊 **Data Requirements for 5-Minute Intervals**

### **Minimum Data Needed:**
- **For testing**: 1 day (288 intervals)
- **For training**: 7 days (2,016 intervals) 
- **Recommended**: 30+ days (8,640+ intervals)
- **Optimal**: 90+ days (25,920+ intervals)

### **Data Quality Tips:**
- ✅ No missing 5-minute intervals
- ✅ Consistent timestamp format
- ✅ Clean OHLCV data
- ✅ Volume data included

## 💡 **Performance Optimization**

### **For Better Results:**

```bash
# Longer sequences for better pattern recognition
--seq_len 576 --pred_len 144  # 48 hours → 12 hours

# Adaptive learning for stability
--loss adaptive_gmadl --beta_start 1.1 --beta_end 1.7

# Smaller batches for high-frequency data
--batch_size 8

# More patience for convergence
--patience 7
```

### **For Faster Training:**

```bash
# Shorter sequences
--seq_len 144 --pred_len 36  # 12 hours → 3 hours

# Fixed GMADL
--loss gmadl --beta 1.5

# Larger batches (if memory allows)
--batch_size 32

# Fewer epochs for testing
--train_epochs 10
```

## 📈 **Expected Results**

### **Training Time (5-minute intervals):**
- **Quick test**: 2-5 minutes
- **Standard training**: 20-45 minutes  
- **Production training**: 1-3 hours

### **Prediction Horizons:**
- **Short-term**: 36 intervals (3 hours)
- **Medium-term**: 72 intervals (6 hours)
- **Long-term**: 144 intervals (12 hours)

### **GMADL vs Other Losses:**
- **vs MSE**: More robust to price spikes
- **vs MAE**: Faster convergence
- **vs Huber**: Better for time series patterns

## 🔍 **Monitoring Training**

### **What to Watch:**
```
🔧 Training Configuration:
  📊 Loss Function: GMADL
  📈 Beta Parameter: 1.5
  ⏱️  Sequence Length: 288 intervals (1440 minutes)
  🎯 Prediction Length: 72 intervals (360 minutes)

>>>>>>>start training : informer_BTC_5min_ftMS_sl288...
Epoch: 1 cost time: 45.2s 
Train Loss: 0.234 Vali Loss: 0.189
```

### **Good Signs:**
- ✅ Validation loss decreasing
- ✅ Training stable (not jumping around)
- ✅ Loss values in reasonable range (0.1-1.0)

### **Warning Signs:**
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ Loss exploding (>10.0)
- ⚠️ No improvement after many epochs

## 🆘 **Troubleshooting**

### **Out of Memory:**
```bash
# Reduce batch size and sequence length
--batch_size 4 --seq_len 144
```

### **Poor Convergence:**
```bash
# Try adaptive GMADL
--loss adaptive_gmadl --beta_start 1.1 --beta_end 1.9
```

### **Overfitting:**
```bash
# Add more regularization
--dropout 0.1 --patience 3
```

### **Training Too Slow:**
```bash
# Use fixed GMADL with higher beta
--loss gmadl --beta 1.7
```

## 🎯 **Advanced Tips**

### **For Different Crypto Types:**

**Bitcoin/Ethereum (Less volatile):**
```bash
--beta 1.7 --seq_len 576 --pred_len 144
```

**Altcoins (More volatile):**
```bash
--beta 1.3 --seq_len 288 --pred_len 72
```

**Meme coins (Very volatile):**
```bash
--beta 1.1 --seq_len 144 --pred_len 36
```

### **For Different Trading Styles:**

**Scalping (minutes):**
```bash
--seq_len 144 --pred_len 12  # 12 hours → 1 hour
```

**Day Trading (hours):**
```bash
--seq_len 288 --pred_len 72  # 24 hours → 6 hours
```

**Swing Trading (days):**
```bash
--seq_len 576 --pred_len 288  # 48 hours → 24 hours
```

## 🚀 **Ready to Start!**

**Your next command should be:**
```bash
python crypto_train_gmadl_5min.py --crypto_data your_5min_data.csv --coin_name BTC --do_predict
```

This will train with optimal GMADL settings for 5-minute crypto prediction! 🎯