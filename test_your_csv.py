#!/usr/bin/env python3
"""
Quick test script to validate your LINK CSV file format
"""
import pandas as pd
import sys

def test_csv_format(csv_file):
    print("🔍 TESTING YOUR LINK CSV FORMAT")
    print("=" * 50)
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"✅ File loaded successfully: {len(df)} rows")
        
        # Show columns
        print(f"\n📋 Your columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Test timestamp parsing
        print(f"\n🕐 Testing timestamp format...")
        print(f"   Sample Open Time: {df['Open Time'].iloc[0]}")
        
        try:
            # Try date string format (2022-10-27)
            date_parsed = pd.to_datetime(df['Open Time'])
            print(f"   ✅ Date string format detected!")
            print(f"   📅 First date: {date_parsed.iloc[0]}")
            print(f"   📅 Last date: {date_parsed.iloc[-1]}")
            
            # Check interval
            if len(df) > 1:
                interval = date_parsed.iloc[1] - date_parsed.iloc[0]
                print(f"   ⏱️  Time interval: {interval}")
                if interval.days == 1:
                    print(f"   ✅ Perfect! Daily intervals detected")
                else:
                    print(f"   ⚠️  Non-daily intervals: {interval}")
        except:
            print(f"   ❌ Could not parse date format")
            return False
        
        # Check required columns
        required_cols = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n❌ Missing required columns: {missing_cols}")
            return False
        else:
            print(f"\n✅ All required columns present!")
        
        # Show sample data
        print(f"\n📊 Sample data (first 3 rows):")
        print(df[required_cols].head(3).to_string())
        
        # Data quality checks
        print(f"\n🔍 Data quality:")
        print(f"   📊 Total rows: {len(df)}")
        print(f"   ✅ Usable rows: {len(df)} (will use first 800)")
        
        # Check for missing values
        missing = df[required_cols].isnull().sum()
        if missing.sum() > 0:
            print(f"   ⚠️  Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"      - {col}: {count}")
        else:
            print(f"   ✅ No missing values in key columns")
        
        # Price analysis
        if 'Close' in df.columns:
            close_prices = df['Close'].dropna()
            print(f"\n💰 LINK Price Analysis:")
            print(f"   📈 Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
            print(f"   💰 Current (last): ${close_prices.iloc[-1]:.2f}")
            volatility = close_prices.std() / close_prices.mean()
            print(f"   📊 Volatility: {volatility:.3f}")
            if volatility > 0.1:
                print(f"   🔥 High volatility - Perfect for advanced models!")
            
        print(f"\n🎉 YOUR CSV IS READY FOR ULTRA-OPTIMIZED TRAINING!")
        print(f"🚀 Next step: python train_link_ultra_optimized.py")
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_your_csv.py your_file.csv")
        print("\nOr just run it and enter the path:")
        csv_file = input("📁 Enter path to your LINK CSV file: ").strip()
    else:
        csv_file = sys.argv[1]
    
    success = test_csv_format(csv_file)
    if success:
        print(f"\n✅ SUCCESS! Your CSV is compatible!")
    else:
        print(f"\n❌ Please fix the issues above and try again.")