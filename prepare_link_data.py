#!/usr/bin/env python3
"""
LINK Data Preparation Script
Handles the exact column format provided by the user.
"""

import pandas as pd
import os
import sys

def prepare_link_data(input_file, output_file):
    """
    Prepare LINK data with exact column names:
    Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Volume, Number of Trades, Taker Buy Base, Taker Buy Quote
    """
    print(f"🔗 Preparing LINK data from: {input_file}")
    print("=" * 50)
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"✅ File loaded successfully")
        print(f"📊 Original shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Display current columns
    print(f"\n📋 Current columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}'")
    
    # Expected exact column names
    expected_columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base', 'Taker Buy Quote'
    ]
    
    # Check if columns match
    if list(df.columns) == expected_columns:
        print(f"\n✅ Perfect! Columns match exactly as expected")
    else:
        print(f"\n⚠️  Column mismatch detected")
        print(f"Expected: {expected_columns}")
        print(f"Got: {list(df.columns)}")
        
        # Try to proceed anyway
        if len(df.columns) >= 5:  # At least OHLC + Volume
            print(f"🔧 Proceeding with available columns...")
        else:
            print(f"❌ Insufficient columns for training")
            return False
    
    # Convert Open Time (milliseconds) to datetime
    print(f"\n🕐 Converting timestamps...")
    df['date'] = pd.to_datetime(df['Open Time'], unit='ms')
    
    # Show timestamp conversion
    print(f"  📅 First timestamp: {df['Open Time'].iloc[0]} -> {df['date'].iloc[0]}")
    print(f"  📅 Last timestamp: {df['Open Time'].iloc[-1]} -> {df['date'].iloc[-1]}")
    
    # Check for 1-day intervals
    if len(df) > 1:
        time_diff = df['date'].iloc[1] - df['date'].iloc[0]
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
        if col in df.columns:
            training_columns.append(col)
    
    # Add Close as the last column (target)
    training_columns.append('Close')
    
    # Select and reorder columns
    df_processed = df[training_columns].copy()
    
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
    print(f"\n🎉 LINK data prepared successfully!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Final shape: {df_processed.shape}")
    print(f"📋 Columns: {list(df_processed.columns)}")
    print(f"📅 Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
    
    # Calculate coverage
    days_count = len(df_processed)
    weeks_count = days_count / 7
    months_count = days_count / 30
    print(f"📈 Coverage: {days_count} days ({weeks_count:.1f} weeks, {months_count:.1f} months)")
    
    # Price volatility check
    if 'Close' in df_processed.columns:
        close_prices = df_processed['Close'].dropna()
        volatility = close_prices.std() / close_prices.mean()
        print(f"💹 LINK price volatility: {volatility:.3f}")
        if volatility > 0.1:
            print(f"  🔥 High volatility - GMADL will be very beneficial!")
        else:
            print(f"  📊 Moderate volatility - Good for GMADL")
    
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python prepare_link_data.py <input_file> <output_file>")
        print("Example: python prepare_link_data.py LINK_raw.csv data/LINK_daily.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print("🔗 LINK Data Preparation Tool")
    print("=" * 50)
    
    success = prepare_link_data(input_file, output_file)
    
    if success:
        print(f"\n✅ Success! Your LINK data is ready for GMADL training!")
        print(f"\n🚀 Next step - validate your data:")
        print(f"python validate_crypto_data_1day.py {output_file}")
        print(f"\n🚀 Then start training:")
        print(f"python crypto_train_gmadl_1day.py --crypto_data {output_file} --coin_name LINK --do_predict")
    else:
        print(f"\n❌ Failed to prepare data. Please check your input file.")
        sys.exit(1)

if __name__ == '__main__':
    main()