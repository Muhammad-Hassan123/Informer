#!/usr/bin/env python3
"""
Crypto Data Validation Script for 1-Day Intervals
This script helps you validate your daily crypto data format before training with GMADL.
"""

import pandas as pd
import sys
import argparse
from datetime import datetime, timedelta

def validate_crypto_data_1day(file_path):
    """Validate 1-day crypto data format and provide feedback"""
    
    print(f"📅 Validating 1-day crypto data: {file_path}")
    print("=" * 60)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully")
        print(f"📊 Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check columns
    print(f"\n📋 Current columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
    
    # Expected columns (flexible matching)
    expected_patterns = {
        'timestamp': ['open time', 'time', 'timestamp', 'date', 'datetime'],
        'open': ['open'],
        'high': ['high'],
        'low': ['low'], 
        'close': ['close'],
        'volume': ['volume', 'vol']
    }
    
    print(f"\n🔍 Column Analysis:")
    found_columns = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern_name, patterns in expected_patterns.items():
            if any(pattern in col_lower for pattern in patterns):
                found_columns[pattern_name] = col
                print(f"  ✅ {pattern_name.title()}: '{col}'")
                break
    
    # Check for missing essential columns
    essential_cols = ['open', 'high', 'low', 'close']
    missing_essential = [col for col in essential_cols if col not in found_columns]
    
    if missing_essential:
        print(f"\n⚠️  Missing essential columns: {missing_essential}")
        print("   These are required for crypto price prediction.")
    else:
        print(f"\n✅ All essential columns found!")
    
    # Check timestamp column and daily intervals
    if 'timestamp' in found_columns:
        ts_col = found_columns['timestamp']
        print(f"\n📅 Timestamp Analysis (Daily Data):")
        print(f"  Column: '{ts_col}'")
        
        # Sample a few values
        sample_values = df[ts_col].head(3).tolist()
        print(f"  Sample values: {sample_values}")
        
        # Try to detect timestamp format
        first_val = df[ts_col].iloc[0]
        if isinstance(first_val, (int, float)):
            if first_val > 1e10:  # Likely milliseconds
                print(f"  ✅ Detected: Millisecond timestamp")
                try:
                    converted = pd.to_datetime(first_val, unit='ms')
                    print(f"  Converts to: {converted}")
                except:
                    print(f"  ⚠️  Could not convert timestamp")
            else:  # Likely seconds
                print(f"  ✅ Detected: Second timestamp") 
                try:
                    converted = pd.to_datetime(first_val, unit='s')
                    print(f"  Converts to: {converted}")
                except:
                    print(f"  ⚠️  Could not convert timestamp")
        else:
            print(f"  ℹ️  String timestamp detected")
            try:
                converted = pd.to_datetime(first_val)
                print(f"  ✅ Converts to: {converted}")
            except:
                print(f"  ⚠️  Could not convert timestamp")
        
        # Check for daily intervals
        if len(df) > 1:
            try:
                if isinstance(df[ts_col].iloc[0], (int, float)):
                    if df[ts_col].iloc[0] > 1e10:
                        dates = pd.to_datetime(df[ts_col], unit='ms')
                    else:
                        dates = pd.to_datetime(df[ts_col], unit='s')
                else:
                    dates = pd.to_datetime(df[ts_col])
                
                time_diff = dates.iloc[1] - dates.iloc[0]
                print(f"  📊 Time interval: {time_diff}")
                
                if time_diff.days == 1:
                    print(f"  ✅ Perfect! Exactly 1-day intervals detected")
                elif time_diff.days == 0 and time_diff.seconds == 86400:
                    print(f"  ✅ Perfect! Exactly 1-day intervals detected")
                else:
                    print(f"  ⚠️  Warning: Not exactly 1-day intervals!")
                    print(f"     Consider resampling your data to daily intervals")
                    
                # Check for gaps
                expected_days = (dates.iloc[-1] - dates.iloc[0]).days + 1
                actual_days = len(df)
                if expected_days != actual_days:
                    missing_days = expected_days - actual_days
                    print(f"  ⚠️  Potential missing days: {missing_days}")
                else:
                    print(f"  ✅ No missing days detected")
                    
            except Exception as e:
                print(f"  ⚠️  Could not analyze time intervals: {e}")
    else:
        print(f"\n⚠️  No timestamp column detected")
        print("   Will create artificial daily timestamps during processing")
    
    # Data quality checks for daily crypto data
    print(f"\n🔍 Data Quality (Daily Crypto):")
    print(f"  Total days: {len(df)}")
    
    # Calculate time coverage
    days_of_data = len(df)
    weeks_of_data = days_of_data / 7
    months_of_data = days_of_data / 30
    years_of_data = days_of_data / 365
    
    print(f"  📈 Coverage: {days_of_data} days = {weeks_of_data:.1f} weeks = {months_of_data:.1f} months = {years_of_data:.1f} years")
    
    # Data coverage assessment
    if days_of_data < 30:
        print(f"  ❌ Too little data ({days_of_data} days). Need at least 30 days.")
    elif days_of_data < 180:
        print(f"  ⚠️  Limited data ({days_of_data} days). 180+ days recommended.")
    elif days_of_data < 365:
        print(f"  ✅ Good data amount ({days_of_data} days)")
    else:
        print(f"  🎉 Excellent data amount ({days_of_data} days)!")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"  ⚠️  Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"    - {col}: {count} missing values")
    else:
        print(f"  ✅ No missing values")
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"  📊 Numeric columns: {len(numeric_cols)}")
    
    if len(numeric_cols) >= 4:  # At least OHLC
        print(f"  ✅ Sufficient numeric data for training")
    else:
        print(f"  ⚠️  Few numeric columns detected")
    
    # Daily data specific checks
    if 'close' in found_columns:
        close_col = found_columns['close']
        close_prices = df[close_col].dropna()
        if len(close_prices) > 0:
            price_volatility = close_prices.std() / close_prices.mean()
            print(f"  📈 Price volatility: {price_volatility:.3f}")
            if price_volatility > 0.1:
                print(f"    🔥 High volatility - GMADL will be very beneficial!")
            elif price_volatility > 0.05:
                print(f"    📊 Medium volatility - Good for GMADL")
            else:
                print(f"    📉 Low volatility - Consider higher beta values")
    
    # Sample data preview
    print(f"\n📋 Data Preview (first 3 rows):")
    print(df.head(3).to_string())
    
    # Recommendations for daily data
    print(f"\n💡 Recommendations for 1-Day Crypto Training:")
    
    if missing_essential:
        print(f"  ❌ Cannot proceed without: {missing_essential}")
        print(f"     Please ensure your data has Open, High, Low, Close columns")
        return False
    else:
        print(f"  ✅ Data format looks excellent for daily GMADL training!")
        
        # Suggest optimal parameters based on data size
        if days_of_data >= 800:
            seq_len = 90
            pred_len = 14
            epochs = 25
            beta = 1.6
            print(f"  🎯 Optimal for long-term analysis (800+ days)")
        elif days_of_data >= 365:
            seq_len = 60
            pred_len = 7
            epochs = 20
            beta = 1.6
            print(f"  🎯 Good for medium-term analysis (1+ year)")
        elif days_of_data >= 180:
            seq_len = 45
            pred_len = 5
            epochs = 15
            beta = 1.5
            print(f"  🎯 Suitable for short-term analysis (6+ months)")
        else:
            seq_len = 30
            pred_len = 3
            epochs = 10
            beta = 1.4
            print(f"  🎯 Basic training possible (limited data)")
        
        # Suggest training command
        print(f"\n🚀 Suggested GMADL training command:")
        print(f"python crypto_train_gmadl_1day.py \\")
        print(f"    --crypto_data {file_path} \\")
        print(f"    --coin_name YOUR_COIN \\")
        print(f"    --seq_len {seq_len} \\")
        print(f"    --pred_len {pred_len} \\")
        print(f"    --loss gmadl \\")
        print(f"    --beta {beta} \\")
        print(f"    --train_epochs {epochs} \\")
        print(f"    --do_predict")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Validate 1-day crypto data for GMADL training')
    parser.add_argument('file_path', help='Path to your daily crypto CSV file')
    
    args = parser.parse_args()
    
    print("📅 Daily Crypto Data Validator for GMADL Training")
    print("=" * 60)
    
    is_valid = validate_crypto_data_1day(args.file_path)
    
    if is_valid:
        print(f"\n🎉 Validation completed successfully!")
        print(f"Your daily crypto data is ready for GMADL training!")
        print(f"🎯 Perfect for swing trading and trend analysis")
    else:
        print(f"\n❌ Validation failed!")
        print(f"Please fix the issues above before training.")
        sys.exit(1)

if __name__ == '__main__':
    main()