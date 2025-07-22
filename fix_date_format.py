#!/usr/bin/env python3
"""
Date Format Fixer for LINK CSV Files
Handles various date formats and converts them to standard format
"""
import pandas as pd
import sys
from datetime import datetime

def detect_and_fix_date_format(csv_file, output_file=None):
    """
    Detect and fix date format in CSV file
    """
    print("🔧 DATE FORMAT FIXER")
    print("=" * 40)
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        print(f"✅ File loaded: {len(df)} rows")
        
        if 'Open Time' not in df.columns:
            print(f"❌ 'Open Time' column not found")
            return False
        
        # Show sample dates
        print(f"\n📅 Sample dates from your file:")
        for i in range(min(5, len(df))):
            print(f"  Row {i+1}: {df['Open Time'].iloc[i]}")
        
        # Try different date formats
        date_formats = [
            ('%Y-%m-%d', 'YYYY-MM-DD (2022-10-27)'),
            ('%Y/%m/%d', 'YYYY/MM/DD (2022/10/27)'),
            ('%d-%m-%Y', 'DD-MM-YYYY (27-10-2022)'),
            ('%d/%m/%Y', 'DD/MM/YYYY (27/10/2022)'),
            ('%m-%d-%Y', 'MM-DD-YYYY (10-27-2022)'),
            ('%m/%d/%Y', 'MM/DD/YYYY (10/27/2022)'),
            ('%b %d, %Y', 'Mon DD, YYYY (Oct 27, 2022)'),
            ('%B %d, %Y', 'Month DD, YYYY (October 27, 2022)'),
        ]
        
        successful_format = None
        parsed_dates = None
        
        # Try parsing as standard pandas datetime first
        try:
            parsed_dates = pd.to_datetime(df['Open Time'])
            successful_format = "Auto-detected"
            print(f"✅ Auto-detected date format successfully!")
        except:
            # Try specific formats
            for fmt, description in date_formats:
                try:
                    parsed_dates = pd.to_datetime(df['Open Time'], format=fmt)
                    successful_format = f"{fmt} ({description})"
                    print(f"✅ Detected format: {description}")
                    break
                except:
                    continue
        
        # Try milliseconds as last resort
        if parsed_dates is None:
            try:
                parsed_dates = pd.to_datetime(df['Open Time'], unit='ms')
                successful_format = "Milliseconds"
                print(f"✅ Detected milliseconds format")
            except:
                print(f"❌ Could not parse any date format")
                return False
        
        # Validate dates
        print(f"\n📊 Date Analysis:")
        print(f"  📅 First date: {parsed_dates.iloc[0]}")
        print(f"  📅 Last date: {parsed_dates.iloc[-1]}")
        print(f"  📊 Total days: {(parsed_dates.iloc[-1] - parsed_dates.iloc[0]).days}")
        
        if len(df) > 1:
            interval = parsed_dates.iloc[1] - parsed_dates.iloc[0]
            print(f"  ⏱️  Interval: {interval}")
            
            if interval.days == 1:
                print(f"  ✅ Daily intervals detected")
            elif interval.seconds == 300:  # 5 minutes
                print(f"  ✅ 5-minute intervals detected")
            elif interval.seconds == 3600:  # 1 hour
                print(f"  ✅ Hourly intervals detected")
            else:
                print(f"  ⚠️  Custom interval: {interval}")
        
        # Create fixed DataFrame
        df_fixed = df.copy()
        df_fixed['Open Time'] = parsed_dates.dt.strftime('%Y-%m-%d')
        
        if 'Close Time' in df_fixed.columns:
            try:
                close_dates = pd.to_datetime(df['Close Time'])
                df_fixed['Close Time'] = close_dates.dt.strftime('%Y-%m-%d')
                print(f"✅ Fixed Close Time format too")
            except:
                print(f"⚠️  Could not fix Close Time format")
        
        # Save fixed file
        if output_file is None:
            output_file = csv_file.replace('.csv', '_fixed.csv')
        
        df_fixed.to_csv(output_file, index=False)
        
        print(f"\n🎉 SUCCESS!")
        print(f"  📁 Original format: {successful_format}")
        print(f"  📁 Fixed file saved: {output_file}")
        print(f"  📊 Sample fixed dates:")
        for i in range(min(3, len(df_fixed))):
            print(f"    {df_fixed['Open Time'].iloc[i]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔧 LINK CSV Date Format Fixer")
    print("=" * 50)
    print("This tool fixes date formats in your LINK CSV file")
    print("Supported formats:")
    print("  • 2022-10-27 (YYYY-MM-DD)")
    print("  • 2022/10/27 (YYYY/MM/DD)")
    print("  • 27-10-2022 (DD-MM-YYYY)")
    print("  • 10/27/2022 (MM/DD/YYYY)")
    print("  • Oct 27, 2022 (Month DD, YYYY)")
    print("  • 1666828800000 (Milliseconds)")
    print("  • And more...")
    print("=" * 50)
    
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        input_file = input("📁 Enter path to your LINK CSV file: ").strip()
    
    if not input_file.endswith('.csv'):
        print("❌ Please provide a CSV file")
        return
    
    # Ask for output file
    output_file = input(f"📁 Output file (press Enter for auto-name): ").strip()
    if not output_file:
        output_file = None
    
    success = detect_and_fix_date_format(input_file, output_file)
    
    if success:
        print(f"\n✅ Date format fixed successfully!")
        print(f"🚀 Now you can use the fixed file for training:")
        print(f"   python3 test_your_csv.py {output_file or input_file.replace('.csv', '_fixed.csv')}")
    else:
        print(f"\n❌ Could not fix date format. Please check your file.")

if __name__ == "__main__":
    main()