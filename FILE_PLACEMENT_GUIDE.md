# 📁 File Placement Guide for Crypto Training

## ✅ **Current Status: All Files Are Already Correctly Placed!**

All the crypto training files have been created in the **root directory** of your Informer project, which is exactly where they should be.

## 📂 **Complete Directory Structure**

```
Informer2020/                          # Your main project folder
├── 🆕 crypto_train.py                 # ✅ Main crypto training script
├── 🆕 validate_crypto_data.py         # ✅ Data validation tool
├── 🆕 run_crypto_example.sh           # ✅ Example training commands
├── 🆕 CRYPTO_TRAINING_README.md       # ✅ Detailed usage guide
├── 🆕 CRYPTO_SETUP_COMPLETE.md        # ✅ Setup summary
├── 🆕 FILE_PLACEMENT_GUIDE.md         # ✅ This guide
│
├── main_informer.py                   # Original Informer training script
├── requirements.txt                   # Python dependencies
├── README.md                          # Original project README
│
├── data/                              # Data directory
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading classes
│   └── 📂 [Your crypto CSV files go here]
│
├── models/                            # Model architecture
│   ├── __init__.py
│   ├── model.py                       # Informer model
│   ├── attn.py                        # Attention mechanisms
│   ├── encoder.py                     # Encoder layers
│   ├── decoder.py                     # Decoder layers
│   └── embed.py                       # Embedding layers
│
├── exp/                               # Experiment management
│   ├── __init__.py
│   ├── exp_basic.py                   # Base experiment class
│   └── exp_informer.py                # Informer experiment class (✅ Modified)
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── tools.py                       # Helper functions
│   ├── metrics.py                     # Evaluation metrics
│   ├── timefeatures.py               # Time feature engineering
│   └── masking.py                     # Attention masking
│
├── scripts/                           # Training scripts for ETT data
│   ├── ETTh1.sh
│   ├── ETTh2.sh
│   ├── ETTm1.sh
│   └── WTH.sh
│
└── checkpoints/                       # Model checkpoints (created during training)
    └── [Training outputs will be saved here]
```

## 🎯 **Where to Put Your Crypto Data**

### Option 1: In the data/ folder (Recommended)
```bash
Informer2020/
├── data/
│   └── your_crypto_data.csv          # 👈 Put your CSV here
```

### Option 2: In the root folder (Also works)
```bash
Informer2020/
├── your_crypto_data.csv              # 👈 Or put it here
```

### Option 3: Anywhere with full path
```bash
/path/to/your/crypto_data.csv         # 👈 Use full path in command
```

## 🚀 **How to Use from Current Location**

Since all files are already in place, you can run commands directly:

### From the Informer2020/ root directory:

```bash
# Validate your data (put CSV in data/ folder)
python validate_crypto_data.py data/your_crypto_data.csv

# Or validate from anywhere
python validate_crypto_data.py /full/path/to/your_crypto_data.csv

# Train the model
python crypto_train.py --crypto_data data/your_crypto_data.csv --coin_name BTC

# Or with full path
python crypto_train.py --crypto_data /full/path/to/your_crypto_data.csv --coin_name BTC
```

## 📋 **Quick Setup Checklist**

- ✅ All crypto training files are in root directory
- ✅ Original Informer files are intact  
- ✅ `exp/exp_informer.py` has been modified to support custom data
- ✅ Scripts are executable (`chmod +x` applied)
- 📁 **Next step**: Put your CSV file in `data/` folder
- 🚀 **Then run**: `python validate_crypto_data.py data/your_file.csv`

## 🔧 **File Permissions**

The executable files have been made runnable:
```bash
✅ crypto_train.py           # Main training script
✅ validate_crypto_data.py   # Data validation script  
✅ run_crypto_example.sh     # Example shell script
```

## 💡 **Pro Tips**

1. **Keep your CSV in `data/` folder** - It's organized and follows the project structure
2. **Use relative paths** - `data/your_file.csv` instead of full paths
3. **All commands run from root** - Stay in the `Informer2020/` directory
4. **Outputs go to `checkpoints/`** - Models and predictions saved automatically

## 🎉 **You're All Set!**

Everything is in the right place! Just add your crypto CSV file and start training:

```bash
# 1. Put your CSV in data/ folder
cp your_crypto_data.csv data/

# 2. Validate it
python validate_crypto_data.py data/your_crypto_data.csv

# 3. Start training!
python crypto_train.py --crypto_data data/your_crypto_data.csv --coin_name BTC
```

**No need to move any files - they're already perfectly placed!** 🚀