from flask import Flask, render_template, request, jsonify, send_file
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import WeatherDataLoader
from models.model_builder import ModelBuilder
from models.model_trainer import ModelTrainer
from utils.visualization import WeatherVisualizer
from utils.helpers import WeatherPredictor, create_sample_forecast_data

app = Flask(__name__)
app.secret_key = 'weather_ai_secret_key'

class WebWeatherApp:
    def __init__(self):
        self.model_builder = ModelBuilder()
        self.model_trainer = ModelTrainer()
        self.data_loader = WeatherDataLoader()
        self.visualizer = WeatherVisualizer()
        self.load_models()
    
    def load_models(self):
        """Load trained models for web interface"""
        try:
            # Load training artifacts
            self.model_trainer.scaler = joblib.load('training_artifacts/scaler.pkl')
            self.model_trainer.feature_columns = joblib.load('training_artifacts/feature_columns.pkl')
            self.model_trainer.results = joblib.load('training_artifacts/training_results.pkl')
            
            # Load best model
            best_model_name, best_model_results = self.model_trainer.get_best_model()
            self.best_model = best_model_results['model']
            
            # Create predictor
            self.predictor = WeatherPredictor(
                self.best_model,
                self.model_trainer.scaler,
                self.model_trainer.feature_columns
            )
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def generate_performance_chart(self):
        """Generate model performance chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            models = list(self.model_trainer.results.keys())
            r2_scores = [self.model_trainer.results[model]['test_metrics']['r2'] for model in models]
            mae_scores = [self.model_trainer.results[model]['test_metrics']['mae'] for model in models]
            
            # R² scores
            bars1 = ax1.bar(models, r2_scores, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#607D8B', '#795548', '#E91E63'])
            ax1.set_title('Model R² Scores', fontsize=14, fontweight='bold')
            ax1.set_ylabel('R² Score')
            ax1.set_ylim(0.9, 1.01)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # MAE scores
            bars2 = ax2.bar(models, mae_scores, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#607D8B', '#795548', '#E91E63'])
            ax2.set_title('Model MAE Scores', fontsize=14, fontweight='bold')
            ax2.set_ylabel('MAE (°C)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}°C', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to bytes
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plt.close()
            
            return base64.b64encode(img.getvalue()).decode()
        except Exception as e:
            print(f"Error generating chart: {e}")
            return None

weather_app = WebWeatherApp()

@app.route('/')
def index():
    """Main dashboard page"""
    best_model_name, best_model_results = weather_app.model_trainer.get_best_model()
    
    # Get model performance metrics
    metrics = {
        'best_model': best_model_name,
        'r2_score': f"{best_model_results['test_metrics']['r2']:.3f}",
        'mae': f"{best_model_results['test_metrics']['mae']:.2f}°C",
        'rmse': f"{best_model_results['test_metrics']['rmse']:.2f}°C"
    }
    
    # Generate performance chart
    chart_url = weather_app.generate_performance_chart()
    
    return render_template('index.html', metrics=metrics, chart_url=chart_url)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for weather prediction"""
    try:
        # Get form data
        data = request.get_json()
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind_speed = float(data['wind_speed'])
        precipitation = float(data.get('precipitation', 0.0))
        
        # Create current conditions (simplified for demo)
        current_conditions = {
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            # Add some default values for other features
            'day_of_year_sin': np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365.25),
            'day_of_year_cos': np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365.25),
            'month_sin': np.sin(2 * np.pi * datetime.now().month / 12),
            'month_cos': np.cos(2 * np.pi * datetime.now().month / 12),
        }
        
        # Make prediction
        predicted_temp = weather_app.predictor.predict(current_conditions)
        
        # Determine weather condition
        if predicted_temp < 10:
            condition = "❄️ Cold"
            bg_color = "#4FC3F7"
        elif predicted_temp < 20:
            condition = "🌤️ Mild" 
            bg_color = "#81C784"
        elif predicted_temp < 30:
            condition = "☀️ Warm"
            bg_color = "#FFB74D"
        else:
            condition = "🔥 Hot"
            bg_color = "#E57373"
        
        return jsonify({
            'success': True,
            'predicted_temperature': f"{predicted_temp:.1f}",
            'condition': condition,
            'bg_color': bg_color,
            'current_conditions': {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'precipitation': precipitation
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/forecast')
def forecast():
    """7-day forecast page"""
    # Generate sample forecast using your existing function
    forecast_df = create_sample_forecast_data(weather_app.predictor, days=7)
    forecast_data = forecast_df.to_dict('records')
    
    return render_template('forecast.html', forecast=forecast_data)

@app.route('/models')
def models():
    """Model performance page"""
    results = weather_app.model_trainer.results
    models_data = []
    
    for model_name, result in results.items():
        models_data.append({
            'name': model_name.replace('_', ' ').title(),
            'r2': f"{result['test_metrics']['r2']:.3f}",
            'mae': f"{result['test_metrics']['mae']:.2f}°C",
            'rmse': f"{result['test_metrics']['rmse']:.2f}°C",
            'color': 'success' if result['test_metrics']['r2'] > 0.99 else 'warning'
        })
    
    # Sort by R² score
    models_data.sort(key=lambda x: float(x['r2']), reverse=True)
    
    return render_template('models.html', models=models_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)