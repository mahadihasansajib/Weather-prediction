import os
import pickle
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import WeatherDataLoader
from models.model_builder import ModelBuilder
from models.model_trainer import ModelTrainer
from utils.visualization import WeatherVisualizer
from utils.helpers import WeatherPredictor, create_sample_forecast_data

class WeatherPredictionApp:
    def __init__(self):
        self.model_builder = ModelBuilder()
        self.model_trainer = ModelTrainer()
        self.data_loader = WeatherDataLoader()
        self.visualizer = WeatherVisualizer()
        self.is_trained = False
        
    def load_saved_models(self):
        """Load previously trained models"""
        print("Loading saved models...")
        try:
            # Check if saved models directory exists
            if not os.path.exists('saved_models'):
                print("Saved models directory not found")
                return False
            
            # Load training artifacts first
            if os.path.exists('training_artifacts'):
                try:
                    self.model_trainer.scaler = joblib.load('training_artifacts/scaler.pkl')
                    self.model_trainer.feature_columns = joblib.load('training_artifacts/feature_columns.pkl')
                    self.model_trainer.results = joblib.load('training_artifacts/training_results.pkl')
                    print("Training artifacts loaded successfully!")
                except Exception as e:
                    print(f"Error loading artifacts: {e}")
                    return False
            
            # Load models using the model_builder
            success = self.model_builder.load_models('saved_models')
            if success:
                self.is_trained = True
                print("All models loaded successfully!")
                return True
            else:
                print("No saved models found. Please train models first.")
                self.is_trained = False
                return False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False
            return False
    
    def show_model_performance(self):
        """Show performance of loaded models"""
        if not self.is_trained:
            print("No trained models available. Please train or load models first.")
            return
        
        self.model_trainer.plot_model_comparison()
        
        best_model_name, best_model_results = self.model_trainer.get_best_model()
        print(f"\nBest Performing Model: {best_model_name}")
        print(f"Test R²: {best_model_results['test_metrics']['r2']:.3f}")
        print(f"Test MAE: {best_model_results['test_metrics']['mae']:.2f}°C")
        print(f"Test RMSE: {best_model_results['test_metrics']['rmse']:.2f}°C")
    
    def make_prediction(self):
        """Make a weather prediction using current conditions"""
        if not self.is_trained:
            print("No trained models available.")
            return
        
        best_model_name, best_model_results = self.model_trainer.get_best_model()
        best_model = best_model_results['model']
        
        # Create predictor
        predictor = WeatherPredictor(
            best_model,
            self.model_trainer.scaler,
            self.model_trainer.feature_columns
        )
        
        # Sample current conditions (you would get these from real-time data)
        current_conditions = {
            'temperature': 18.5,
            'humidity': 62.0,
            'pressure': 1015.2,
            'precipitation': 0.0,
            'wind_speed': 12.3,
            'day_of_year_sin': np.sin(2 * np.pi * 105 / 365.25),  # Day 105
            'day_of_year_cos': np.cos(2 * np.pi * 105 / 365.25),
            'month_sin': np.sin(2 * np.pi * 4 / 12),  # April
            'month_cos': np.cos(2 * np.pi * 4 / 12),
            'temperature_lag_1': 17.8,
            'temperature_lag_2': 17.2,
            'temperature_lag_3': 16.9,
            'temperature_lag_7': 16.5,
            'humidity_lag_1': 65.0,
            'pressure_lag_1': 1014.8,
            'temperature_rolling_mean_7': 17.2,
            'temperature_rolling_std_7': 1.1,
            'pressure_rolling_mean_7': 1014.5,
            'temp_humidity_interaction': 18.5 * 62.0,
            'temp_pressure_interaction': 18.5 * 1015.2,
            'temp_trend_3d': 18.5 - 16.9,
            'day_of_week': 2,  # Tuesday
            'week_of_year': 15
        }
        
        # Make prediction
        predicted_temp = predictor.predict(current_conditions)
        print(f"\nCurrent conditions:")
        print(f"  Temperature: {current_conditions['temperature']}°C")
        print(f"  Humidity: {current_conditions['humidity']}%")
        print(f"  Pressure: {current_conditions['pressure']} hPa")
        print(f"\nPredicted temperature for tomorrow: {predicted_temp:.1f}°C")
        
        return predicted_temp, current_conditions, predictor
    
    def create_forecast(self, days=7):
        """Create multi-day forecast"""
        if not self.is_trained:
            print("No trained models available.")
            return
        
        _, current_conditions, predictor = self.make_prediction()
        
        print(f"\nCreating {days}-day forecast...")
        forecast_df = create_sample_forecast_data(
            predictor, 
            days=days, 
            initial_conditions=current_conditions
        )
        
        print("\nWeather Forecast:")
        print("="*50)
        for _, row in forecast_df.iterrows():
            print(f"Day {row['day']}: {row['predicted_temperature']:.1f}°C, "
                  f"Humidity: {row['humidity']:.0f}%, "
                  f"Pressure: {row['pressure']:.1f} hPa")
        
        return forecast_df
    
    def run_demo(self):
        """Run a complete demo of the weather prediction system"""
        print("=== Weather Prediction AI Demo ===")
        
        # Try to load existing models
        if not self.load_saved_models():
            print("\nNo pre-trained models found.")
            print("Please run 'python train_model.py' first to train models.")
            return
        
        print("\n1. Model Performance Overview:")
        self.show_model_performance()
        
        print("\n2. Single Day Prediction:")
        self.make_prediction()
        
        print("\n3. 7-Day Forecast:")
        forecast_df = self.create_forecast(days=7)
        
        print("\n=== Demo Completed ===")
        
        return forecast_df

def main():
    app = WeatherPredictionApp()
    app.run_demo()

if __name__ == "__main__":
    main()