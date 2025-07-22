#!/usr/bin/env python3
"""
Quick fix for Windows path issues in train_link_ultra_optimized.py
"""
import os

def fix_windows_paths():
    print("🔧 FIXING WINDOWS PATHS IN TRAINING SCRIPT")
    print("=" * 50)
    
    # Read the current file
    with open('train_link_ultra_optimized.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix path separators for Windows
    fixes = [
        ("'./checkpoints/'", "'checkpoints'"),
        ("'./data/LINK_ultra_optimized.csv'", "'data/LINK_ultra_optimized.csv'"),
        ("f'{results_dir}pred.npy'", "f'{results_dir}pred.npy'.replace('/', os.sep)"),
        ("f'{results_dir}true.npy'", "f'{results_dir}true.npy'.replace('/', os.sep)"),
    ]
    
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Fixed: {old} → {new}")
    
    # Add os import at the top if not present
    if "import os" not in content:
        content = content.replace("import torch", "import os\nimport torch")
        print("✅ Added: import os")
    
    # Write back the fixed file
    with open('train_link_ultra_optimized.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n🎉 Windows paths fixed!")
    print("🚀 Now run the training again!")

if __name__ == "__main__":
    fix_windows_paths()