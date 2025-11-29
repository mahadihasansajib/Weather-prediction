#!/usr/bin/env python3
"""
Setup script for the Weather Prediction AI Project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.6.0",
        "xgboost>=1.5.0",
        "shap>=0.40.0",
        "streamlit>=1.0.0",
        "plotly>=5.0.0",
        "joblib>=1.0.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nAll packages installed successfully!")

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data',
        'models', 
        'utils',
        'notebooks',
        'tests',
        'saved_models',
        'training_artifacts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")
    
    # Create __init__.py files
    for dir in ['data', 'models', 'utils', 'tests']:
        init_file = os.path.join(dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write('')
        print(f"✓ Created {init_file}")

def main():
    print("=== Weather Prediction AI Project Setup ===")
    
    create_directory_structure()
    print()
    install_requirements()
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Run 'python train_model.py' to train the models")
    print("2. Run 'python main.py' to see the demo")
    print("3. Check the notebooks/ directory for exploration notebooks")

if __name__ == "__main__":
    main()