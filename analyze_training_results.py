#!/usr/bin/env python3
"""
Generic Training Results Analyzer
Automatically detects and analyzes your actual training results
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from datetime import datetime

def find_checkpoint_folders():
    """Find all checkpoint folders from training runs"""
    checkpoint_pattern = os.path.join('checkpoints', 'informer_*')
    folders = glob.glob(checkpoint_pattern)
    return folders

def extract_model_info_from_path(folder_path):
    """Extract model configuration from folder path"""
    folder_name = os.path.basename(folder_path)
    
    # Extract model name
    if 'ensemble_model_1' in folder_name:
        model_name = 'Model 1 (Standard)'
        architecture = '2E-1D-512dim-8heads'
    elif 'ensemble_model_2' in folder_name:
        model_name = 'Model 2 (Deeper)'
        architecture = '3E-2D-512dim-8heads'
    elif 'ensemble_model_3' in folder_name:
        model_name = 'Model 3 (Wider)'
        architecture = '2E-1D-768dim-12heads'
    else:
        model_name = 'Unknown Model'
        architecture = 'Unknown'
    
    # Extract sequence parameters
    seq_match = re.search(r'sl(\d+)_ll(\d+)_pl(\d+)', folder_name)
    if seq_match:
        seq_len, label_len, pred_len = seq_match.groups()
        sequence_info = f'{seq_len}→{pred_len} days'
    else:
        sequence_info = 'Unknown sequence'
    
    return model_name, architecture, sequence_info

def read_training_metrics(folder_path):
    """Try to read training metrics from various sources"""
    
    # Try to find metrics.txt or similar files
    metrics_files = [
        os.path.join(folder_path, 'metrics.txt'),
        os.path.join(folder_path, 'results.txt'),
        os.path.join(folder_path, 'training_log.txt')
    ]
    
    mse, mae = None, None
    
    # Try to read from metrics files
    for metrics_file in metrics_files:
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    content = f.read()
                    
                # Look for MSE and MAE patterns
                mse_match = re.search(r'mse[:\s]+([0-9.]+)', content, re.IGNORECASE)
                mae_match = re.search(r'mae[:\s]+([0-9.]+)', content, re.IGNORECASE)
                
                if mse_match:
                    mse = float(mse_match.group(1))
                if mae_match:
                    mae = float(mae_match.group(1))
                    
                if mse and mae:
                    break
            except:
                continue
    
    # Try to read from numpy files
    if mse is None or mae is None:
        try:
            pred_file = os.path.join(folder_path, 'pred.npy')
            true_file = os.path.join(folder_path, 'true.npy')
            
            if os.path.exists(pred_file) and os.path.exists(true_file):
                pred = np.load(pred_file)
                true = np.load(true_file)
                
                # Calculate metrics
                mse = np.mean((pred - true) ** 2)
                mae = np.mean(np.abs(pred - true))
        except:
            pass
    
    return mse, mae

def get_performance_rating(mse, mae):
    """Get performance rating based on metrics"""
    if mse is None or mae is None:
        return "Unknown", "Unknown"
    
    if mse < 0.5:
        rating = "🔥 EXCELLENT"
        confidence = "95%+"
    elif mse < 1.0:
        rating = "🟢 VERY GOOD"
        confidence = "85-95%"
    elif mse < 2.0:
        rating = "🟡 GOOD"
        confidence = "75-85%"
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
        confidence = "<75%"
    
    return rating, confidence

def analyze_data_info():
    """Analyze processed data information"""
    data_file = os.path.join('data', 'LINK_ultra_optimized.csv')
    
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            n_features = len(df.columns) - 1  # Exclude date column
            n_rows = len(df)
            date_range = f"{df['date'].min()} to {df['date'].max()}"
            
            # Count technical indicators (features beyond basic OHLCV)
            basic_features = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            technical_indicators = n_features - len(basic_features) + 1  # +1 because we exclude date
            
            return {
                'features': n_features,
                'rows': n_rows,
                'date_range': date_range,
                'technical_indicators': technical_indicators
            }
        except:
            pass
    
    return None

def main():
    print("🔍 GENERIC TRAINING RESULTS ANALYZER")
    print("=" * 80)
    print("📊 Automatically detecting your actual training results...")
    print()
    
    # Find checkpoint folders
    checkpoint_folders = find_checkpoint_folders()
    
    if not checkpoint_folders:
        print("❌ No checkpoint folders found!")
        print("💡 Make sure you have run training and have folders in 'checkpoints/' directory")
        return
    
    print(f"✅ Found {len(checkpoint_folders)} trained model(s)")
    print()
    
    # Analyze each model
    results = []
    total_mse = 0
    total_mae = 0
    valid_models = 0
    
    print("🏆 INDIVIDUAL MODEL PERFORMANCE:")
    print("-" * 80)
    
    for i, folder in enumerate(checkpoint_folders, 1):
        model_name, architecture, sequence_info = extract_model_info_from_path(folder)
        mse, mae = read_training_metrics(folder)
        rating, confidence = get_performance_rating(mse, mae)
        
        print(f"{model_name}:")
        print(f"  📁 Folder: {os.path.basename(folder)}")
        print(f"  📐 Architecture: {architecture}")
        print(f"  🎯 Sequence: {sequence_info}")
        
        if mse is not None and mae is not None:
            print(f"  📊 MSE: {mse:.4f} | MAE: {mae:.4f}")
            print(f"  🎯 Rating: {rating}")
            print(f"  🔮 Confidence: {confidence}")
            
            total_mse += mse
            total_mae += mae
            valid_models += 1
            
            results.append({
                'Model': model_name,
                'Architecture': architecture,
                'Sequence': sequence_info,
                'MSE': mse,
                'MAE': mae,
                'Rating': rating.replace('🔥 ', '').replace('🟢 ', '').replace('🟡 ', '').replace('🔴 ', ''),
                'Confidence': confidence,
                'Folder': os.path.basename(folder)
            })
        else:
            print(f"  ❌ No metrics found")
            print(f"  💡 Try running the model again or check for results files")
        
        print()
    
    # Overall performance summary
    if valid_models > 0:
        avg_mse = total_mse / valid_models
        avg_mae = total_mae / valid_models
        overall_rating, _ = get_performance_rating(avg_mse, avg_mae)
        
        print("🏆 ENSEMBLE PERFORMANCE SUMMARY:")
        print("=" * 50)
        print(f"📊 Valid Models: {valid_models}/{len(checkpoint_folders)}")
        print(f"📊 Average MSE: {avg_mse:.4f}")
        print(f"📊 Average MAE: {avg_mae:.4f}")
        print(f"🎯 Overall Rating: {overall_rating}")
        
        # Practical interpretation
        print(f"\n💡 PRACTICAL INTERPRETATION:")
        print(f"📊 Your MAE of {avg_mae:.2f} means:")
        print(f"   💰 Predictions are typically within ${avg_mae:.2f} of actual LINK price")
        print(f"   📈 For a $15 LINK price, expect ±{(avg_mae/15)*100:.1f}% accuracy")
        
        if avg_mae < 0.8:
            quality = "EXCELLENT"
        elif avg_mae < 1.2:
            quality = "VERY GOOD"
        elif avg_mae < 2.0:
            quality = "GOOD"
        else:
            quality = "FAIR"
        
        print(f"   🎯 This is {quality} for crypto prediction!")
        
        # Best model
        if results:
            best_model = min(results, key=lambda x: x['MSE'])
            print(f"\n🏆 BEST PERFORMING MODEL:")
            print(f"   🥇 {best_model['Model']} - MSE: {best_model['MSE']:.4f}")
            print(f"   📐 Architecture: {best_model['Architecture']}")
    
    # Data information
    data_info = analyze_data_info()
    if data_info:
        print(f"\n📊 TRAINING DATA ANALYSIS:")
        print(f"  📈 Total Features: {data_info['features']}")
        print(f"  🔧 Technical Indicators: {data_info['technical_indicators']}")
        print(f"  📊 Data Points: {data_info['rows']}")
        print(f"  📅 Date Range: {data_info['date_range']}")
    
    # Save results
    if results:
        try:
            df = pd.DataFrame(results)
            os.makedirs('checkpoints', exist_ok=True)
            summary_file = os.path.join('checkpoints', 'GENERIC_TRAINING_ANALYSIS.csv')
            df.to_csv(summary_file, index=False)
            
            print(f"\n💾 RESULTS SAVED:")
            print(f"  📊 Analysis: {summary_file}")
            
            # Save detailed summary
            with open(os.path.join('checkpoints', 'TRAINING_SUMMARY.txt'), 'w') as f:
                f.write(f"Training Results Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Models Analyzed: {valid_models}/{len(checkpoint_folders)}\n")
                if valid_models > 0:
                    f.write(f"Average MSE: {avg_mse:.4f}\n")
                    f.write(f"Average MAE: {avg_mae:.4f}\n")
                    f.write(f"Overall Rating: {overall_rating}\n")
                    f.write(f"Prediction Accuracy: ±${avg_mae:.2f}\n")
                
                if data_info:
                    f.write(f"\nData Information:\n")
                    f.write(f"Features: {data_info['features']}\n")
                    f.write(f"Technical Indicators: {data_info['technical_indicators']}\n")
                    f.write(f"Rows: {data_info['rows']}\n")
            
            print(f"  📄 Summary: checkpoints/TRAINING_SUMMARY.txt")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    print(f"\n🎉 ANALYSIS COMPLETED!")
    if valid_models > 0:
        print(f"🔥 Your models are performing {'EXCELLENTLY' if avg_mse < 1.0 else 'VERY WELL'}!")
    else:
        print(f"💡 Run your training first to generate results to analyze!")

if __name__ == '__main__':
    main()