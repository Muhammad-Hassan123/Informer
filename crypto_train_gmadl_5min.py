#!/usr/bin/env python3
"""
Crypto Training with GMADL Loss on 5-minute intervals
Specialized script for high-frequency crypto price prediction using GMADL loss function.
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from exp.exp_informer import Exp_Informer

def prepare_crypto_data_5min(csv_file, output_file):
    """
    Prepare crypto data for 5-minute interval training
    Expected columns: Open Time | Open | High | Low | Close | Volume | Close Time | Quote Asset Volume | Number of Trades | Taker Buy Base Volume | Taker Buy Quote Volume
    """
    print(f"🕐 Reading 5-minute crypto data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Rename columns to match expected format
    expected_columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                       'Taker Buy Base Volume', 'Taker Buy Quote Volume']
    
    if len(df.columns) == len(expected_columns):
        df.columns = expected_columns
    
    # Convert Open Time to datetime (assuming it's in milliseconds timestamp)
    if 'Open Time' in df.columns:
        # If it's timestamp in milliseconds
        if df['Open Time'].iloc[0] > 1e10:
            df['date'] = pd.to_datetime(df['Open Time'], unit='ms')
        else:
            df['date'] = pd.to_datetime(df['Open Time'], unit='s')
    else:
        # Create a 5-minute interval date column if not present
        df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='5min')
    
    # Select relevant features for 5-minute training
    feature_columns = ['date', 'Open', 'High', 'Low', 'Volume', 'Quote Asset Volume', 
                      'Number of Trades', 'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Close']
    
    # Ensure all feature columns exist
    available_columns = ['date']
    for col in ['Open', 'High', 'Low', 'Volume', 'Close']:
        if col in df.columns:
            available_columns.append(col)
    
    # Add additional columns if they exist
    for col in ['Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume', 'Taker Buy Quote Volume']:
        if col in df.columns:
            available_columns.append(col)
    
    # Make sure Close is the last column (target)
    if 'Close' in available_columns:
        available_columns.remove('Close')
        available_columns.append('Close')
    
    df_processed = df[available_columns].copy()
    
    # Sort by date
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    
    # Verify 5-minute intervals
    if len(df_processed) > 1:
        time_diff = df_processed['date'].iloc[1] - df_processed['date'].iloc[0]
        print(f"📊 Detected time interval: {time_diff}")
        if time_diff.total_seconds() != 300:  # 5 minutes = 300 seconds
            print(f"⚠️  Warning: Data may not be exactly 5-minute intervals!")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    print(f"✅ Processed 5-minute data saved to {output_file}")
    print(f"📊 Data shape: {df_processed.shape}")
    print(f"📋 Columns: {list(df_processed.columns)}")
    print(f"📅 Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    print(f"⏱️  Total time span: {df_processed['date'].max() - df_processed['date'].min()}")
    
    # Calculate some 5-minute specific stats
    hours_of_data = len(df_processed) / 12  # 12 intervals per hour
    days_of_data = hours_of_data / 24
    print(f"📈 Data coverage: {hours_of_data:.1f} hours ({days_of_data:.1f} days)")
    
    return len(available_columns) - 1  # Subtract 1 for date column

def main():
    parser = argparse.ArgumentParser(description='[Informer] Crypto Forecasting with GMADL Loss on 5-min intervals')
    
    # Data arguments
    parser.add_argument('--crypto_data', type=str, required=True, help='path to your 5-minute crypto CSV file')
    parser.add_argument('--coin_name', type=str, default='CRYPTO', help='name of your cryptocurrency')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='informer', help='model of experiment, options: [informer, informerstack]')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; MS recommended for price prediction')
    parser.add_argument('--target', type=str, default='Close', help='target feature (Close price)')
    parser.add_argument('--freq', type=str, default='5min', help='freq for time features encoding (5min intervals)')
    
    # Training arguments optimized for 5-minute intervals
    parser.add_argument('--seq_len', type=int, default=288, help='input sequence length (288 * 5min = 24 hours)')
    parser.add_argument('--label_len', type=int, default=144, help='start token length (144 * 5min = 12 hours)')
    parser.add_argument('--pred_len', type=int, default=72, help='prediction sequence length (72 * 5min = 6 hours)')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    
    # GMADL Loss function parameters
    parser.add_argument('--loss', type=str, default='gmadl', help='loss function: mse, mae, gmadl, adaptive_gmadl, weighted_gmadl')
    parser.add_argument('--beta', type=float, default=1.5, help='beta parameter for GMADL loss (1.0=MAE, 2.0=MSE, 1.5=balanced)')
    parser.add_argument('--beta_start', type=float, default=1.2, help='starting beta for adaptive GMADL')
    parser.add_argument('--beta_end', type=float, default=1.8, help='ending beta for adaptive GMADL')
    parser.add_argument('--weight_decay_loss', type=float, default=0.95, help='weight decay for weighted GMADL')
    
    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (smaller for 5-min high-freq data)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    
    # Other arguments
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict future data')
    
    args = parser.parse_args()
    
    print("🪙 Crypto Training with GMADL Loss on 5-minute intervals")
    print("=" * 60)
    
    # Prepare data
    processed_data_path = f'./data/{args.coin_name}_5min_processed.csv'
    num_features = prepare_crypto_data_5min(args.crypto_data, processed_data_path)
    
    # Set up data parameters
    args.data = f'{args.coin_name}_5min'
    args.root_path = './data/'
    args.data_path = f'{args.coin_name}_5min_processed.csv'
    
    # Set input/output dimensions based on features
    if args.features == 'M':  # Multivariate
        args.enc_in = num_features
        args.dec_in = num_features  
        args.c_out = num_features
    elif args.features == 'MS':  # Multivariate to univariate (recommended)
        args.enc_in = num_features
        args.dec_in = num_features
        args.c_out = 1
    else:  # Univariate
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
    
    # Other required parameters
    args.padding = 0
    args.distil = True
    args.mix = True
    args.output_attention = False
    args.inverse = False
    args.use_amp = False
    args.num_workers = 0
    args.des = f'gmadl_5min_beta{args.beta}'
    args.lradj = 'type1'
    args.use_multi_gpu = False
    args.devices = '0'
    args.cols = None
    
    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    print(f"\n🔧 Training Configuration:")
    print(f"  📊 Loss Function: {args.loss.upper()}")
    if 'gmadl' in args.loss.lower():
        print(f"  📈 Beta Parameter: {args.beta}")
        if args.loss.lower() == 'adaptive_gmadl':
            print(f"  📉 Beta Range: {args.beta_start} → {args.beta_end}")
    print(f"  ⏱️  Sequence Length: {args.seq_len} intervals ({args.seq_len * 5} minutes)")
    print(f"  🎯 Prediction Length: {args.pred_len} intervals ({args.pred_len * 5} minutes)")
    print(f"  🎲 Features: {args.features}")
    print(f"  📦 Batch Size: {args.batch_size}")
    print(f"  🔄 Epochs: {args.train_epochs}")
    print(f"  💻 Device: {'GPU' if args.use_gpu else 'CPU'}")
    
    print(f"\n🚀 Starting Training...")
    print('Args in experiment:')
    print(args)
    
    # Run experiments
    Exp = Exp_Informer
    
    for ii in range(args.itr):
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
            args.model, args.data, args.features, 
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, 
            args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()

    print("🎉 Training completed!")
    print(f"📁 Results saved in: ./checkpoints/{setting}/")
    
    # Print GMADL loss explanation
    print(f"\n💡 GMADL Loss Information:")
    print(f"  📊 Beta = {args.beta}")
    if args.beta == 1.0:
        print(f"  📈 Equivalent to MAE (Mean Absolute Error)")
    elif args.beta == 2.0:
        print(f"  📈 Equivalent to MSE (Mean Squared Error)")
    else:
        print(f"  📈 Balanced between MAE and MSE - robust to outliers")
    print(f"  🎯 Optimized for crypto price prediction on 5-minute intervals")

if __name__ == '__main__':
    main()