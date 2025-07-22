#!/usr/bin/env python3
"""
Quick CSV test script for Windows - LINK data validation
"""
import pandas as pd
import sys
import os

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
        except Exception as e:
            print(f"   ❌ Could not parse date format: {e}")
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
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

if __name__ == "__main__":
    print("🔧 LINK CSV Tester for Windows")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("📁 Enter path to your LINK CSV file: ").strip()
        # Remove quotes if user copied path with quotes
        csv_file = csv_file.strip('"').strip("'")
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    success = test_csv_format(csv_file)
    if success:
        print(f"\n✅ SUCCESS! Your CSV is ready for training!")
        print(f"🚀 Next: Run train_link_ultra_optimized.py")
    else:
        print(f"\n❌ Please fix the issues above and try again.")
    
    input("\nPress Enter to exit...")