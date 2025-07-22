#!/usr/bin/env python3
"""
Quick Windows path fix for train_link_ultra_optimized.py
"""
import os
import re

def fix_windows_paths():
    print("🔧 FIXING WINDOWS PATHS - FINAL FIX")
    print("=" * 50)
    
    try:
        # Read the current file
        with open('train_link_ultra_optimized.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply all Windows path fixes
        fixes_applied = 0
        
        # Fix 1: Results directory paths
        if "results_dir = f'./checkpoints/{setting}/'" in content:
            content = content.replace(
                "results_dir = f'./checkpoints/{setting}/'",
                "results_dir = os.path.join('checkpoints', setting)"
            )
            fixes_applied += 1
            print("✅ Fixed results_dir path")
        
        # Fix 2: pred.npy loading
        if "f'{results_dir}pred.npy'" in content:
            content = content.replace(
                "f'{results_dir}pred.npy'",
                "os.path.join(results_dir, 'pred.npy')"
            )
            fixes_applied += 1
            print("✅ Fixed pred.npy path")
        
        # Fix 3: true.npy loading  
        if "f'{results_dir}true.npy'" in content:
            content = content.replace(
                "f'{results_dir}true.npy'",
                "os.path.join(results_dir, 'true.npy')"
            )
            fixes_applied += 1
            print("✅ Fixed true.npy path")
        
        # Fix 4: CSV results file
        if "'./checkpoints/LINK_ULTRA_OPTIMIZED_RESULTS.csv'" in content:
            content = content.replace(
                "'./checkpoints/LINK_ULTRA_OPTIMIZED_RESULTS.csv'",
                "os.path.join('checkpoints', 'LINK_ULTRA_OPTIMIZED_RESULTS.csv')"
            )
            fixes_applied += 1
            print("✅ Fixed CSV results path")
        
        # Fix 5: PNG plot file
        if "'./checkpoints/LINK_ULTRA_OPTIMIZED_ANALYSIS.png'" in content:
            content = content.replace(
                "'./checkpoints/LINK_ULTRA_OPTIMIZED_ANALYSIS.png'",
                "os.path.join('checkpoints', 'LINK_ULTRA_OPTIMIZED_ANALYSIS.png')"
            )
            fixes_applied += 1
            print("✅ Fixed PNG plot path")
        
        # Add os import if not present
        if "import os" not in content and fixes_applied > 0:
            content = re.sub(r'(import torch)', r'import os\n\1', content)
            print("✅ Added import os")
        
        # Write back the fixed file
        with open('train_link_ultra_optimized.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n🎉 Applied {fixes_applied} Windows path fixes!")
        print("🚀 Now run the training again - it should work completely!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please apply the fixes manually.")

if __name__ == "__main__":
    fix_windows_paths()