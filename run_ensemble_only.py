#!/usr/bin/env python3
"""
Run only the ensemble analysis using existing prediction files
"""
import os
import numpy as np

def run_ensemble_analysis():
    """Run ensemble analysis using existing prediction files"""
    
    print("🚀 Running Ensemble Analysis with Existing Predictions")
    print("=" * 60)
    
    # Model settings that were already trained
    model_settings = [
        'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_1',
        'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_2', 
        'informer_LINK_ULTRA_ftMS_sl90_ll45_pl10_ensemble_model_3'
    ]
    
    ensemble_predictions = []
    ensemble_actuals = []
    
    # Load all existing predictions
    for i, setting in enumerate(model_settings):
        print(f"\n📊 Loading Model {i+1} predictions...")
        
        results_dir = os.path.join('results', setting)
        pred_file = os.path.join(results_dir, 'pred.npy')
        true_file = os.path.join(results_dir, 'true.npy')
        
        if os.path.exists(pred_file) and os.path.exists(true_file):
            pred = np.load(pred_file)
            true = np.load(true_file)
            
            ensemble_predictions.append(pred)
            if i == 0:  # Same actuals for all models
                ensemble_actuals = true
                
            print(f"✅ Model {i+1} loaded - Shape: {pred.shape}")
        else:
            print(f"❌ Model {i+1} files not found")
            return False
    
    if len(ensemble_predictions) == 0:
        print("❌ No predictions found!")
        return False
        
    print(f"\n🎉 Successfully loaded {len(ensemble_predictions)} models!")
    
    # Convert to numpy arrays
    ensemble_predictions = np.array(ensemble_predictions)
    
    # Calculate ensemble statistics
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    std_predictions = np.std(ensemble_predictions, axis=0)
    
    # Get the last prediction for each model (most recent forecast)
    last_predictions = ensemble_predictions[:, -1, :, -1]  # (n_models, pred_len)
    
    print(f"\n📈 Ensemble Results:")
    print(f"  📊 Ensemble shape: {ensemble_predictions.shape}")
    print(f"  📊 Mean prediction shape: {mean_predictions.shape}")
    print(f"  📊 Last predictions shape: {last_predictions.shape}")
    
    # Calculate confidence intervals
    confidence_95 = 1.96 * std_predictions  # 95% confidence interval
    upper_bound = mean_predictions + confidence_95
    lower_bound = mean_predictions - confidence_95
    
    # Calculate ensemble metrics
    ensemble_mse = np.mean((mean_predictions - ensemble_actuals) ** 2)
    ensemble_mae = np.mean(np.abs(mean_predictions - ensemble_actuals))
    
    print(f"\n📊 Ensemble Performance:")
    print(f"  🎯 Ensemble MSE: {ensemble_mse:.6f}")
    print(f"  🎯 Ensemble MAE: {ensemble_mae:.6f}")
    
    # Show latest predictions with confidence
    print(f"\n🔮 Latest 10-day Forecast (with 95% confidence):")
    latest_mean = last_predictions.mean(axis=0)
    latest_std = last_predictions.std(axis=0)
    latest_upper = latest_mean + 1.96 * latest_std
    latest_lower = latest_mean - 1.96 * latest_std
    
    for day in range(len(latest_mean)):
        print(f"  Day {day+1}: {latest_mean[day]:.4f} [{latest_lower[day]:.4f}, {latest_upper[day]:.4f}]")
    
    # Save ensemble results
    results_dir = 'results/ensemble_analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, 'ensemble_mean.npy'), mean_predictions)
    np.save(os.path.join(results_dir, 'ensemble_std.npy'), std_predictions)
    np.save(os.path.join(results_dir, 'ensemble_upper.npy'), upper_bound)
    np.save(os.path.join(results_dir, 'ensemble_lower.npy'), lower_bound)
    np.save(os.path.join(results_dir, 'latest_forecast.npy'), latest_mean)
    
    print(f"\n💾 Ensemble results saved to: {results_dir}")
    print("✅ Ensemble analysis completed successfully!")
    
    return True

if __name__ == "__main__":
    run_ensemble_analysis()