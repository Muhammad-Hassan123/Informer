#!/usr/bin/env python3
"""
Crypto Data Validation Script
This script helps you validate your crypto data format before training.
"""

import pandas as pd
import sys
import argparse
from datetime import datetime

def validate_crypto_data(file_path):
    """Validate crypto data format and provide feedback"""
    
    print(f"🔍 Validating crypto data: {file_path}")
    print("=" * 50)
    
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
    
    # Check timestamp column
    if 'timestamp' in found_columns:
        ts_col = found_columns['timestamp']
        print(f"\n📅 Timestamp Analysis:")
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
    else:
        print(f"\n⚠️  No timestamp column detected")
        print("   Will create artificial timestamps during processing")
    
    # Data quality checks
    print(f"\n🔍 Data Quality:")
    print(f"  Total rows: {len(df)}")
    
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
    
    # Sample data preview
    print(f"\n📋 Data Preview (first 3 rows):")
    print(df.head(3).to_string())
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if len(df) < 100:
        print(f"  ⚠️  Very small dataset ({len(df)} rows). Consider getting more data.")
    elif len(df) < 500:
        print(f"  ⚠️  Small dataset ({len(df)} rows). More data would improve results.")
    else:
        print(f"  ✅ Good dataset size ({len(df)} rows)")
    
    if missing_essential:
        print(f"  ❌ Cannot proceed without: {missing_essential}")
        print(f"     Please ensure your data has Open, High, Low, Close columns")
        return False
    else:
        print(f"  ✅ Data format looks good for training!")
        
        # Suggest training command
        print(f"\n🚀 Suggested training command:")
        print(f"python crypto_train.py \\")
        print(f"    --crypto_data {file_path} \\")
        print(f"    --coin_name YOUR_COIN \\")
        print(f"    --features MS \\")
        print(f"    --seq_len 96 \\")
        print(f"    --pred_len 24 \\")
        print(f"    --train_epochs 10")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Validate crypto data for Informer training')
    parser.add_argument('file_path', help='Path to your crypto CSV file')
    
    args = parser.parse_args()
    
    print("🪙 Crypto Data Validator for Informer Model")
    print("=" * 50)
    
    is_valid = validate_crypto_data(args.file_path)
    
    if is_valid:
        print(f"\n🎉 Validation completed successfully!")
        print(f"Your data is ready for training with the Informer model.")
    else:
        print(f"\n❌ Validation failed!")
        print(f"Please fix the issues above before training.")
        sys.exit(1)

if __name__ == '__main__':
    main()