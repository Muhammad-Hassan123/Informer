#!/usr/bin/env python3
"""
Clear explanation of LINK price prediction results
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def explain_prediction_results():
    """Explain the prediction results in clear, understandable terms"""
    
    print("🔍 CHAINLINK (LINK) PRICE PREDICTION RESULTS EXPLAINED")
    print("=" * 70)
    
    # Load the processed data to understand the context
    try:
        data = pd.read_csv('data/LINK_ultra_optimized.csv')
        data['date'] = pd.to_datetime(data['date'])
        
        # Get the last known date and price
        last_date = data['date'].iloc[-1]
        last_price = data['Close'].iloc[-1]
        
        print(f"📅 Your data goes up to: {last_date.strftime('%Y-%m-%d')}")
        print(f"💰 Last known LINK price: ${last_price:.2f}")
        
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        return
    
    # Load ensemble results
    try:
        ensemble_dir = 'results/ensemble_analysis'
        latest_forecast = np.load(os.path.join(ensemble_dir, 'latest_forecast.npy'))
        
        # Calculate confidence intervals from individual models
        model_settings = [
            'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_1',
            'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_2', 
            'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_3'
        ]
        
        # Load individual model predictions for the last forecast
        individual_forecasts = []
        for setting in model_settings:
            pred_file = os.path.join('results', setting, 'pred.npy')
            if os.path.exists(pred_file):
                pred = np.load(pred_file)
                # Get the last prediction (most recent 10-day forecast)
                last_pred = pred[-1, :, -1]  # Last sample, all days, last feature (Close price)
                individual_forecasts.append(last_pred)
        
        individual_forecasts = np.array(individual_forecasts)
        
    except Exception as e:
        print(f"❌ Could not load ensemble results: {e}")
        return
    
    print(f"\n🔮 WHAT THESE PREDICTIONS MEAN:")
    print("-" * 40)
    print(f"• Your AI models analyzed LINK price patterns from {data['date'].iloc[0].strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"• They learned from {len(data)} days of price data")
    print(f"• 3 different AI models made predictions and we combined them for better accuracy")
    print(f"• Each prediction shows the expected LINK price for the next 10 days")
    
    print(f"\n📊 10-DAY LINK PRICE FORECAST:")
    print("=" * 50)
    
    for day in range(len(latest_forecast)):
        # Calculate the future date
        future_date = last_date + timedelta(days=day + 1)
        
        # Get individual model predictions for this day
        day_predictions = individual_forecasts[:, day]
        mean_price = np.mean(day_predictions)
        std_price = np.std(day_predictions)
        
        # 95% confidence interval
        lower_bound = mean_price - 1.96 * std_price
        upper_bound = mean_price + 1.96 * std_price
        
        # Calculate percentage change from last known price
        pct_change = ((mean_price - last_price) / last_price) * 100
        
        print(f"📅 {future_date.strftime('%Y-%m-%d')} (Day {day+1}):")
        print(f"   💰 Predicted Price: ${mean_price:.2f}")
        print(f"   📈 Change from today: {pct_change:+.1f}%")
        print(f"   🎯 95% Confidence Range: ${lower_bound:.2f} - ${upper_bound:.2f}")
        print(f"   📊 Model Agreement: {3 - len(set(np.round(day_predictions, 1)))}/3 models agree")
        print()
    
    # Summary statistics
    avg_predicted_price = np.mean(latest_forecast)
    total_change = ((avg_predicted_price - last_price) / last_price) * 100
    
    print(f"📈 SUMMARY:")
    print("-" * 20)
    print(f"💰 Current LINK Price: ${last_price:.2f}")
    print(f"🔮 Average 10-day Price: ${avg_predicted_price:.2f}")
    print(f"📊 Expected Change: {total_change:+.1f}%")
    
    if total_change > 5:
        print("🚀 Models predict LINK will RISE significantly!")
    elif total_change > 0:
        print("📈 Models predict LINK will rise modestly")
    elif total_change > -5:
        print("📉 Models predict LINK will decline modestly")
    else:
        print("⚠️ Models predict LINK will FALL significantly")
    
    print(f"\n⚠️ IMPORTANT DISCLAIMERS:")
    print("-" * 25)
    print("• These are AI predictions, not financial advice")
    print("• Crypto markets are highly volatile and unpredictable")
    print("• Past performance doesn't guarantee future results")
    print("• Only invest what you can afford to lose")
    print("• Always do your own research before making investment decisions")
    
    # Model performance context
    try:
        # Load test metrics to show model accuracy
        metrics_files = []
        for setting in model_settings:
            metrics_file = os.path.join('results', setting, 'metrics.npy')
            if os.path.exists(metrics_file):
                metrics = np.load(metrics_file)
                metrics_files.append(metrics)
        
        if metrics_files:
            avg_mae = np.mean([m[0] for m in metrics_files])  # MAE is first metric
            print(f"\n🎯 MODEL ACCURACY:")
            print(f"   Average prediction error: ±${avg_mae:.2f}")
            print(f"   (Based on historical testing)")
    except:
        pass

if __name__ == "__main__":
    explain_prediction_results()