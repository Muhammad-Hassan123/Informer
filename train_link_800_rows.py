#!/usr/bin/env python3
"""
LINK Training on First 800 Rows with Confidence Predictions
Specialized script for training on exactly 800 rows and generating 5-10 day predictions with confidence levels.
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from exp.exp_informer import Exp_Informer

def prepare_link_800_rows(csv_file, output_file):
    """
    Prepare exactly first 800 rows of LINK data
    """
    print(f"🔗 Preparing first 800 rows from: {csv_file}")
    print("=" * 60)
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ File loaded successfully")
        print(f"📊 Total rows available: {len(df)}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Take exactly first 800 rows
    if len(df) < 800:
        print(f"❌ Error: File has only {len(df)} rows, need at least 800")
        return False
    
    df_800 = df.head(800).copy()
    print(f"✅ Selected first 800 rows")
    
    # Display current columns
    print(f"\n📋 Current columns:")
    for i, col in enumerate(df_800.columns):
        print(f"  {i+1}. '{col}'")
    
    # Expected exact column names
    expected_columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base', 'Taker Buy Quote'
    ]
    
    # Convert Open Time (milliseconds) to datetime
    print(f"\n🕐 Converting timestamps...")
    df_800['date'] = pd.to_datetime(df_800['Open Time'], unit='ms')
    
    # Show timestamp range
    print(f"  📅 First date: {df_800['date'].iloc[0]}")
    print(f"  📅 Last date: {df_800['date'].iloc[-1]}")
    print(f"  📅 Date range: {(df_800['date'].iloc[-1] - df_800['date'].iloc[0]).days + 1} days")
    
    # Check for 1-day intervals
    if len(df_800) > 1:
        time_diff = df_800['date'].iloc[1] - df_800['date'].iloc[0]
        print(f"  ⏱️  Time interval: {time_diff}")
        if time_diff.days == 1:
            print(f"  ✅ Perfect! 1-day intervals confirmed")
        else:
            print(f"  ⚠️  Warning: Not exactly 1-day intervals")
    
    # Prepare columns for training (Close as target, last column)
    training_columns = ['date', 'Open', 'High', 'Low', 'Volume']
    
    # Add additional columns if they exist
    additional_cols = ['Quote Asset Volume', 'Number of Trades', 'Taker Buy Base', 'Taker Buy Quote']
    for col in additional_cols:
        if col in df_800.columns:
            training_columns.append(col)
    
    # Add Close as the last column (target)
    training_columns.append('Close')
    
    # Select and reorder columns
    df_processed = df_800[training_columns].copy()
    
    # Sort by date to ensure proper order
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    
    # Check for missing values
    missing_counts = df_processed.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠️  Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    - {col}: {count} missing")
    else:
        print(f"\n✅ No missing values detected")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n🎉 LINK 800-row data prepared successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Final shape: {df_processed.shape}")
    print(f"📋 Columns: {list(df_processed.columns)}")
    print(f"📅 Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    
    # Price analysis
    if 'Close' in df_processed.columns:
        close_prices = df_processed['Close'].dropna()
        print(f"💰 LINK price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
        print(f"💰 Current price (last): ${close_prices.iloc[-1]:.2f}")
        volatility = close_prices.std() / close_prices.mean()
        print(f"📈 Price volatility: {volatility:.3f}")
    
    return True

def calculate_prediction_confidence(predictions, actuals=None):
    """
    Calculate confidence levels for predictions
    """
    pred_array = np.array(predictions)
    
    # Calculate prediction variance (uncertainty measure)
    if len(pred_array.shape) > 1:
        pred_variance = np.var(pred_array, axis=0)
        pred_mean = np.mean(pred_array, axis=0)
    else:
        pred_variance = np.var(pred_array)
        pred_mean = np.mean(pred_array)
    
    # Calculate confidence as inverse of normalized variance
    confidence = 1.0 / (1.0 + pred_variance / (pred_mean**2 + 1e-8))
    
    # Normalize confidence to 0-100%
    confidence_pct = confidence * 100
    
    return confidence_pct, pred_variance

def main():
    print("🔗 LINK Training on First 800 Rows with Confidence Predictions")
    print("=" * 70)
    
    # Get input file from user
    input_file = input("📁 Enter path to your LINK CSV file: ").strip()
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    # Prepare data
    print(f"\n🔧 Step 1: Preparing first 800 rows...")
    processed_data_path = './data/LINK_800_rows.csv'
    success = prepare_link_800_rows(input_file, processed_data_path)
    
    if not success:
        print(f"❌ Failed to prepare data")
        return
    
    # Training configuration
    print(f"\n🔧 Step 2: Setting up training configuration...")
    
    class Args:
        def __init__(self):
            # Data parameters
            self.crypto_data = processed_data_path
            self.coin_name = 'LINK_800'
            self.data = 'LINK_800'
            self.root_path = './data/'
            self.data_path = 'LINK_800_rows.csv'
            
            # Model parameters
            self.model = 'informer'
            self.features = 'MS'  # Multivariate to univariate (all features -> Close price)
            self.target = 'Close'
            self.freq = 'd'
            
            # Sequence parameters (optimized for 800 rows)
            self.seq_len = 60    # 60 days history (2 months)
            self.label_len = 30  # 30 days start token
            self.pred_len = 10   # 10 days prediction (max range you wanted)
            
            # Model architecture
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 2048
            self.factor = 5
            self.dropout = 0.05
            self.attn = 'prob'
            self.embed = 'timeF'
            self.activation = 'gelu'
            
            # GMADL Loss parameters
            self.loss = 'adaptive_gmadl'
            self.beta = 1.6
            self.beta_start = 1.4
            self.beta_end = 1.8
            
            # Training parameters
            self.train_epochs = 20  # Sufficient for 800 rows
            self.batch_size = 16    # Smaller batch for limited data
            self.learning_rate = 0.0001
            self.patience = 5
            self.itr = 1
            
            # Technical parameters
            self.enc_in = 9  # Will be set based on data
            self.dec_in = 9
            self.c_out = 1   # Predicting Close price only
            self.padding = 0
            self.distil = True
            self.mix = True
            self.output_attention = False
            self.inverse = False
            self.use_amp = False
            self.num_workers = 0
            self.des = 'link_800_confidence'
            self.lradj = 'type1'
            self.use_multi_gpu = False
            self.devices = '0'
            self.cols = None
            self.checkpoints = './checkpoints/'
            
            # GPU setup
            self.use_gpu = True if torch.cuda.is_available() else False
            self.gpu = 0
            self.do_predict = True
    
    args = Args()
    
    print(f"✅ Configuration ready:")
    print(f"  📊 Training on: 800 rows")
    print(f"  🎯 Prediction length: {args.pred_len} days")
    print(f"  📈 Sequence length: {args.seq_len} days") 
    print(f"  🔄 Epochs: {args.train_epochs}")
    print(f"  💻 Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    # Start training
    print(f"\n🚀 Step 3: Starting training...")
    
    Exp = Exp_Informer
    
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_0'.format(
        args.model, args.data, args.features, 
        args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, 
        args.attn, args.factor, args.embed, args.distil, args.mix, args.des)

    exp = Exp(args)
    
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    test_results = exp.test(setting)
    
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, True)
    
    # Load and analyze results
    print(f"\n📊 Step 4: Analyzing results and calculating confidence...")
    
    results_dir = f'./checkpoints/{setting}/'
    
    try:
        # Load predictions and actual values
        predictions = np.load(f'{results_dir}pred.npy')
        actual_values = np.load(f'{results_dir}true.npy')
        
        print(f"✅ Results loaded successfully")
        print(f"📊 Predictions shape: {predictions.shape}")
        print(f"📊 Actual values shape: {actual_values.shape}")
        
        # Get the last prediction (most recent 10 days forecast)
        last_prediction = predictions[-1, :, -1]  # Last sample, all time steps, Close price
        last_actual = actual_values[-1, :, -1] if actual_values.shape[1] >= args.pred_len else None
        
        # Calculate confidence levels
        confidence_scores, pred_variance = calculate_prediction_confidence(predictions[:, :, -1])
        
        # Load processed data to get actual dates
        df_processed = pd.read_csv(processed_data_path)
        last_date = pd.to_datetime(df_processed['date'].iloc[-1])
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(args.pred_len)]
        
        print(f"\n🎯 LINK Price Predictions (Next {args.pred_len} Days):")
        print("=" * 60)
        print(f"📅 Prediction starts from: {future_dates[0].strftime('%Y-%m-%d')}")
        print(f"📅 Last training date: {last_date.strftime('%Y-%m-%d')}")
        print(f"💰 Current LINK price: ${df_processed['Close'].iloc[-1]:.2f}")
        print()
        
        # Display predictions with confidence
        for i in range(args.pred_len):
            date = future_dates[i]
            price = last_prediction[i]
            confidence = confidence_scores[i] if hasattr(confidence_scores, '__len__') else confidence_scores
            
            # Confidence level interpretation
            if confidence >= 80:
                conf_level = "🟢 HIGH"
            elif confidence >= 60:
                conf_level = "🟡 MEDIUM"
            else:
                conf_level = "🔴 LOW"
            
            print(f"Day {i+1:2d} ({date.strftime('%Y-%m-%d')}): ${price:7.2f} | Confidence: {confidence:5.1f}% {conf_level}")
        
        # Summary statistics
        print(f"\n📈 Prediction Summary:")
        print(f"  💰 Price range: ${last_prediction.min():.2f} - ${last_prediction.max():.2f}")
        print(f"  📊 Average confidence: {np.mean(confidence_scores):.1f}%")
        print(f"  🎯 Trend: {'📈 Upward' if last_prediction[-1] > last_prediction[0] else '📉 Downward'}")
        
        # Calculate percentage changes
        current_price = df_processed['Close'].iloc[-1]
        day_5_change = ((last_prediction[4] - current_price) / current_price) * 100
        day_10_change = ((last_prediction[9] - current_price) / current_price) * 100
        
        print(f"\n📊 Expected Price Changes:")
        print(f"  📅 Day 5:  {day_5_change:+6.2f}% (${last_prediction[4]:.2f})")
        print(f"  📅 Day 10: {day_10_change:+6.2f}% (${last_prediction[9]:.2f})")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'Day': range(1, args.pred_len + 1),
            'Predicted_Price': last_prediction,
            'Confidence_Percent': confidence_scores if hasattr(confidence_scores, '__len__') else [confidence_scores] * args.pred_len,
            'Change_from_Current': [(p - current_price) / current_price * 100 for p in last_prediction]
        })
        
        results_file = f'{results_dir}link_predictions_with_confidence.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices (last 30 days)
        historical_prices = df_processed['Close'].tail(30).values
        historical_dates = pd.to_datetime(df_processed['date'].tail(30))
        
        plt.subplot(2, 1, 1)
        plt.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical LINK Prices')
        plt.plot(future_dates, last_prediction, 'r--', linewidth=2, label='Predictions')
        plt.title('LINK Price: Historical vs Predictions')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot confidence levels
        plt.subplot(2, 1, 2)
        colors = ['green' if c >= 80 else 'orange' if c >= 60 else 'red' for c in confidence_scores]
        plt.bar(range(1, args.pred_len + 1), confidence_scores, color=colors, alpha=0.7)
        plt.title('Prediction Confidence Levels')
        plt.xlabel('Day')
        plt.ylabel('Confidence (%)')
        plt.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_file = f'{results_dir}link_predictions_plot.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_file}")
        
        print(f"\n🎉 Training and prediction completed successfully!")
        print(f"📁 All results saved in: {results_dir}")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()