import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from real_data_processor import RealSolarDataProcessor
import warnings
warnings.filterwarnings('ignore')

class RealSolarPowerPredictor:
    def __init__(self, model_path="real_solar_power_model.pkl"):
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        self.features = []
        self.feature_importance = None
        
    def train_model(self, X, y, use_grid_search=False):
        """Train the XGBoost model with real data"""
        if X is None or y is None or len(X) == 0:
            print("No training data available")
            return False
            
        print(f"Training model with {len(X)} samples and {X.shape[1]} features")
        
        self.features = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        if use_grid_search:
            # Use GridSearchCV for hyperparameter tuning
            print("Performing grid search for optimal parameters...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default parameters
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate the model
        self.evaluate_model(X_train, y_train, X_test, y_test)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save the model
        self.save_model()
        
        # Create visualizations
        self.create_visualizations(X_test, y_test)
        
        return True
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """Evaluate the trained model"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Training predictions
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test predictions
        y_test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Training Performance:")
        print(f"  RMSE: {train_rmse:.4f} kW")
        print(f"  MAE:  {train_mae:.4f} kW")
        print(f"  R²:   {train_r2:.4f}")
        
        print(f"\nTest Performance:")
        print(f"  RMSE: {test_rmse:.4f} kW")
        print(f"  MAE:  {test_mae:.4f} kW")
        print(f"  R²:   {test_r2:.4f}")
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            print(f"\nCross-validation RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        except Exception as e:
            print(f"Cross-validation failed: {e}")
        
        # Performance interpretation
        print(f"\nPerformance Interpretation:")
        if test_r2 > 0.8:
            print("  ✅ Excellent model performance (R² > 0.8)")
        elif test_r2 > 0.6:
            print("  ✅ Good model performance (R² > 0.6)")
        elif test_r2 > 0.4:
            print("  ⚠️  Moderate model performance (R² > 0.4)")
        else:
            print("  ❌ Poor model performance (R² < 0.4)")
        
        if test_rmse < 0.5:
            print("  ✅ Low prediction error (< 0.5 kW)")
        elif test_rmse < 1.0:
            print("  ✅ Moderate prediction error (< 1.0 kW)")
        else:
            print("  ⚠️  High prediction error (> 1.0 kW)")
    
    def create_visualizations(self, X_test, y_test):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        try:
            # Predictions vs Actual
            y_pred = self.model.predict(X_test)
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Predictions vs Actual
            plt.subplot(2, 3, 1)
            plt.scatter(y_test, y_pred, alpha=0.6, s=20)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Power (kW)')
            plt.ylabel('Predicted Power (kW)')
            plt.title('Actual vs Predicted Solar Power')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Residuals
            plt.subplot(2, 3, 2)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Power (kW)')
            plt.ylabel('Residuals (kW)')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Feature Importance
            plt.subplot(2, 3, 3)
            if self.feature_importance is not None:
                top_features = self.feature_importance.head(10)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Feature Importance')
                plt.grid(True, alpha=0.3)
            
            # Plot 4: Power distribution
            plt.subplot(2, 3, 4)
            plt.hist(y_test, bins=50, alpha=0.7, label='Actual', density=True)
            plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
            plt.xlabel('Power (kW)')
            plt.ylabel('Density')
            plt.title('Power Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Hourly patterns
            plt.subplot(2, 3, 5)
            if 'hour' in X_test.columns:
                hourly_actual = y_test.groupby(X_test['hour']).mean()
                hourly_pred = pd.Series(y_pred, index=X_test.index).groupby(X_test['hour']).mean()
                plt.plot(hourly_actual.index, hourly_actual.values, 'o-', label='Actual', alpha=0.7)
                plt.plot(hourly_pred.index, hourly_pred.values, 's-', label='Predicted', alpha=0.7)
                plt.xlabel('Hour of Day')
                plt.ylabel('Average Power (kW)')
                plt.title('Hourly Power Pattern')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Plot 6: Monthly patterns
            plt.subplot(2, 3, 6)
            if 'month' in X_test.columns:
                monthly_actual = y_test.groupby(X_test['month']).mean()
                monthly_pred = pd.Series(y_pred, index=X_test.index).groupby(X_test['month']).mean()
                plt.plot(monthly_actual.index, monthly_actual.values, 'o-', label='Actual', alpha=0.7)
                plt.plot(monthly_pred.index, monthly_pred.values, 's-', label='Predicted', alpha=0.7)
                plt.xlabel('Month')
                plt.ylabel('Average Power (kW)')
                plt.title('Monthly Power Pattern')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('real_model_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Visualization saved as 'real_model_analysis.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def predict(self, X_new):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise Exception("Model not trained. Please train the model first.")
        
        # Ensure the input features match the training features
        missing_cols = set(self.features) - set(X_new.columns)
        if missing_cols:
            print(f"Warning: Missing features {missing_cols}, filling with zeros")
            for col in missing_cols:
                X_new[col] = 0
        
        # Reorder columns to match training
        X_new = X_new[self.features]
        
        return self.model.predict(X_new)
    
    def predict_daily_forecast(self, date=None):
        """Generate a daily power forecast"""
        if not self.is_trained:
            raise Exception("Model not trained. Please train the model first.")
        
        if date is None:
            date = pd.Timestamp.now().date()
        
        # Create hourly data for the day
        hours = range(24)
        forecast_data = []
        
        for hour in hours:
            # Create features for this hour
            features = {
                'hour': hour,
                'month': date.month,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'ghi_normalized': 0.8 if 6 <= hour <= 18 else 0.0,  # Simplified
                'temp_normalized': 0.5,  # Default
                'is_daylight': 1 if 6 <= hour <= 18 else 0,
                'irradiance_ratio': 0.8,  # Default
                'direct_diffuse_ratio': 1.0,  # Default
                'zenith_angle': abs(90 - (hour - 12) * 15),  # Simplified
                'ghi_lag1': 0.0,  # Default
                'temp_lag1': 0.5,  # Default
                'ghi_3h_avg': 0.8 if 6 <= hour <= 18 else 0.0,  # Simplified
                'temp_3h_avg': 0.5  # Default
            }
            
            forecast_data.append(features)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_data)
        
        # Make predictions
        predictions = self.predict(forecast_df)
        
        # Create forecast result
        forecast_result = []
        for i, hour in enumerate(hours):
            forecast_result.append({
                'hour': hour,
                'predicted_power': float(predictions[i]),
                'timestamp': f"{date} {hour:02d}:00:00"
            })
        
        return forecast_result
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'features': self.features,
                'feature_importance': self.feature_importance
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print("No trained model to save")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.features = model_data['features']
            self.feature_importance = model_data.get('feature_importance')
            self.is_trained = True
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}")

def main():
    """Main function to train the model with real data"""
    print("="*60)
    print("REAL SOLAR POWER PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Initialize data processor
    processor = RealSolarDataProcessor()
    
    # Load and combine weather data
    print("\n1. Loading weather data...")
    weather_data = processor.combine_weather_data()
    
    if weather_data is None or len(weather_data) == 0:
        print("❌ No weather data available for training")
        return
    
    print(f"✅ Loaded {len(weather_data)} weather records")
    
    # Load power generation data
    print("\n2. Loading power generation data...")
    power_data = processor.load_power_generation_data()
    
    if power_data is not None:
        print(f"✅ Loaded {len(power_data)} power records")
    else:
        print("⚠️  No power generation data available")
    
    # Preprocess data for training
    print("\n3. Preprocessing data for training...")
    X, y = processor.preprocess_for_training()
    
    if X is None or y is None:
        print("❌ Failed to prepare training data")
        return
    
    print(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train the model
    print("\n4. Training XGBoost model...")
    predictor = RealSolarPowerPredictor()
    
    # Use grid search for better performance (optional)
    use_grid_search = False  # Set to True for hyperparameter tuning
    success = predictor.train_model(X, y, use_grid_search=use_grid_search)
    
    if success:
        print("\n✅ Model training completed successfully!")
        
        # Test daily forecast
        print("\n5. Testing daily forecast...")
        try:
            forecast = predictor.predict_daily_forecast()
            print(f"✅ Generated forecast for {len(forecast)} hours")
            
            # Show sample forecast
            print("\nSample forecast (first 6 hours):")
            for i in range(6):
                hour_data = forecast[i]
                print(f"  {hour_data['hour']:02d}:00 - {hour_data['predicted_power']:.2f} kW")
        except Exception as e:
            print(f"⚠️  Forecast test failed: {e}")
        
        print(f"\n📊 Model saved to: {predictor.model_path}")
        print(f"📈 Analysis saved to: real_model_analysis.png")
        
    else:
        print("❌ Model training failed")

if __name__ == "__main__":
    main()
