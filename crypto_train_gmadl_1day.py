#!/usr/bin/env python3
"""
Crypto Training with GMADL Loss on 1-day intervals
Specialized script for long-term crypto price prediction using GMADL loss function.
Perfect for swing trading and long-term trend analysis.
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from exp.exp_informer import Exp_Informer

def prepare_crypto_data_1day(csv_file, output_file):
    """
    Prepare crypto data for 1-day interval training
    Expected columns: Open Time | Open | High | Low | Close | Volume | Close Time | Quote Asset Volume | Number of Trades | Taker Buy Base Volume | Taker Buy Quote Volume
    """
    print(f"📅 Reading 1-day crypto data from {csv_file}...")
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
        # Create a 1-day interval date column if not present
        df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='1D')
    
    # Select relevant features for 1-day training
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
    
    # Verify 1-day intervals
    if len(df_processed) > 1:
        time_diff = df_processed['date'].iloc[1] - df_processed['date'].iloc[0]
        print(f"📊 Detected time interval: {time_diff}")
        if time_diff.days != 1:
            print(f"⚠️  Warning: Data may not be exactly 1-day intervals!")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df_processed.to_csv(output_file, index=False)
    
    print(f"✅ Processed 1-day data saved to {output_file}")
    print(f"📊 Data shape: {df_processed.shape}")
    print(f"📋 Columns: {list(df_processed.columns)}")
    print(f"📅 Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    print(f"⏱️  Total time span: {df_processed['date'].max() - df_processed['date'].min()}")
    
    # Calculate some 1-day specific stats
    days_of_data = len(df_processed)
    weeks_of_data = days_of_data / 7
    months_of_data = days_of_data / 30
    print(f"📈 Data coverage: {days_of_data} days ({weeks_of_data:.1f} weeks, {months_of_data:.1f} months)")
    
    return len(available_columns) - 1  # Subtract 1 for date column

def main():
    parser = argparse.ArgumentParser(description='[Informer] Crypto Forecasting with GMADL Loss on 1-day intervals')
    
    # Data arguments
    parser.add_argument('--crypto_data', type=str, required=True, help='path to your 1-day crypto CSV file')
    parser.add_argument('--coin_name', type=str, default='CRYPTO', help='name of your cryptocurrency')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='informer', help='model of experiment, options: [informer, informerstack]')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; MS recommended for price prediction')
    parser.add_argument('--target', type=str, default='Close', help='target feature (Close price)')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding (daily intervals)')
    
    # Training arguments optimized for 1-day intervals
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length (60 days = ~2 months)')
    parser.add_argument('--label_len', type=int, default=30, help='start token length (30 days = 1 month)')
    parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length (7 days = 1 week)')
    
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
    parser.add_argument('--beta', type=float, default=1.6, help='beta parameter for GMADL loss (1.6 good for daily data)')
    parser.add_argument('--beta_start', type=float, default=1.3, help='starting beta for adaptive GMADL')
    parser.add_argument('--beta_end', type=float, default=1.9, help='ending beta for adaptive GMADL')
    parser.add_argument('--weight_decay_loss', type=float, default=0.9, help='weight decay for weighted GMADL')
    
    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (larger for daily data)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    
    # Other arguments
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict future data')
    
    args = parser.parse_args()
    
    print("🪙 Crypto Training with GMADL Loss on 1-day intervals")
    print("=" * 60)
    
    # Prepare data
    processed_data_path = f'./data/{args.coin_name}_1day_processed.csv'
    num_features = prepare_crypto_data_1day(args.crypto_data, processed_data_path)
    
    # Set up data parameters
    args.data = f'{args.coin_name}_1day'
    args.root_path = './data/'
    args.data_path = f'{args.coin_name}_1day_processed.csv'
    
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
    args.des = f'gmadl_1day_beta{args.beta}'
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
    print(f"  ⏱️  Sequence Length: {args.seq_len} days ({args.seq_len/7:.1f} weeks)")
    print(f"  🎯 Prediction Length: {args.pred_len} days ({args.pred_len} days ahead)")
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
    
    # Print GMADL loss explanation for daily data
    print(f"\n💡 GMADL Loss Information for Daily Data:")
    print(f"  📊 Beta = {args.beta}")
    if args.beta == 1.0:
        print(f"  📈 Equivalent to MAE (Mean Absolute Error)")
    elif args.beta == 2.0:
        print(f"  📈 Equivalent to MSE (Mean Squared Error)")
    else:
        print(f"  📈 Optimized for daily crypto trends - balances robustness and convergence")
    print(f"  🎯 Perfect for swing trading and long-term trend analysis")
    print(f"  📅 Captures weekly and monthly patterns in crypto markets")

if __name__ == '__main__':
    main()