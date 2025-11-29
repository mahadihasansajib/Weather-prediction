#!/usr/bin/env python3
"""
Main script to train weather prediction models
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import WeatherDataLoader
from models.model_builder import ModelBuilder
from models.model_trainer import ModelTrainer
from utils.visualization import WeatherVisualizer
from utils.helpers import save_project_artifacts
import pandas as pd

def main():
    print("=== Weather Prediction AI Project ===")
    print("Starting model training process...\n")
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    data_loader = WeatherDataLoader()
    
    # Try to load from CSV, or generate sample data
    raw_data = data_loader.load_from_csv('data/weather_data.csv')
    if raw_data is None:
        raw_data = data_loader.generate_sample_data()
        data_loader.save_to_csv('data/weather_data.csv')
    
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Data range: {raw_data['date'].min()} to {raw_data['date'].max()}")
    
    # Step 2: Feature engineering
    print("\nStep 2: Feature engineering...")
    engineered_data = data_loader.get_feature_engineered_data()
    print(f"Engineered data shape: {engineered_data.shape}")
    
    # Step 3: Visualize data
    print("\nStep 3: Visualizing data...")
    visualizer = WeatherVisualizer()
    visualizer.plot_weather_timeseries(raw_data)
    visualizer.plot_correlation_heatmap(engineered_data)
    visualizer.plot_seasonal_patterns(raw_data)
    
    # Step 4: Build models
    print("\nStep 4: Building models...")
    model_builder = ModelBuilder()
    models = model_builder.build_sklearn_models()
    
    # Add neural network
    input_dim = len([col for col in engineered_data.columns 
                    if col not in ['date', 'temperature']])
    model_builder.build_neural_network(input_dim, model_type='simple')
    
    print(f"Built {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    # Step 5: Train models
    print("\nStep 5: Training models...")
    model_trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(
        engineered_data, target_column='temperature'
    )
    
    results = model_trainer.train_models(models, X_train, X_test, y_train, y_test)
    
    # Step 6: Evaluate results
    print("\nStep 6: Model evaluation...")
    model_trainer.plot_model_comparison()
    
    # Show predictions for best model
    best_model_name, best_model_results = model_trainer.get_best_model()
    print(f"\nBest model: {best_model_name}")
    print(f"Test R²: {best_model_results['test_metrics']['r2']:.3f}")
    print(f"Test MAE: {best_model_results['test_metrics']['mae']:.2f}°C")
    
    model_trainer.plot_predictions_vs_actual(best_model_name, y_test)
    
    # Step 7: Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        print(f"\nFeature importance for {best_model_name}:")
        visualizer.plot_feature_importance(
            best_model_results['model'],
            model_trainer.feature_columns,
            top_n=15
        )
    
    # Step 8: Save everything
    print("\nStep 7: Saving models and artifacts...")
    model_builder.save_models('saved_models')
    model_trainer.save_training_artifacts('training_artifacts')
    save_project_artifacts(model_builder, model_trainer, data_loader)
    
    print("\n=== Training completed successfully! ===")
    print(f"Best model: {best_model_name}")
    print(f"Final test R²: {best_model_results['test_metrics']['r2']:.3f}")
    print(f"Final test MAE: {best_model_results['test_metrics']['mae']:.2f}°C")
    
    return model_builder, model_trainer, data_loader

if __name__ == "__main__":
    model_builder, model_trainer, data_loader = main()