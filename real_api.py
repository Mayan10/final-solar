from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import joblib
from real_model_trainer import RealSolarPowerPredictor
from real_data_processor import RealSolarDataProcessor
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Real Solar Power Prediction API",
    description="AI-powered solar energy prediction using real weather data",
    version="2.0.0"
)

# Enable CORS (useful during local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor = None
data_processor = None

# Pydantic models for API
class PowerPredictionRequest(BaseModel):
    hour: int
    month: int
    temperature: float
    global_irradiance: float
    wind_speed: Optional[float] = 0.0
    humidity: Optional[float] = 50.0

class PowerPredictionResponse(BaseModel):
    predicted_power: float
    confidence: float
    timestamp: str

class DailyForecastResponse(BaseModel):
    forecast: List[dict]
    total_daily_power: float
    peak_power: float
    peak_hour: int

class ApplianceControl(BaseModel):
    appliance_id: str
    action: str  # "on" or "off"

class ApplianceStatus(BaseModel):
    id: str
    name: str
    status: str
    consumption: float
    priority: int
    controllable: bool
    priority_level: str

class DigitalTwinState(BaseModel):
    appliances: List[ApplianceStatus]
    total_consumption: float
    available_power: float

# Digital Twin state
digital_twin_state = {
    "appliances": [
        {"id": "ac1", "name": "Air Conditioner 1", "status": "off", "consumption": 2.5, "priority": 1, "controllable": True, "priority_level": "High"},
        {"id": "ac2", "name": "Air Conditioner 2", "status": "off", "consumption": 2.5, "priority": 1, "controllable": True, "priority_level": "High"},
        {"id": "lights", "name": "LED Lights", "status": "off", "consumption": 0.5, "priority": 2, "controllable": True, "priority_level": "Medium"},
        {"id": "tv", "name": "TV", "status": "off", "consumption": 0.3, "priority": 3, "controllable": True, "priority_level": "Low"},
        {"id": "refrigerator", "name": "Refrigerator", "status": "on", "consumption": 0.2, "priority": 1, "controllable": False, "priority_level": "High"},
        {"id": "water_heater", "name": "Water Heater", "status": "off", "consumption": 1.5, "priority": 2, "controllable": True, "priority_level": "Medium"},
        {"id": "washing_machine", "name": "Washing Machine", "status": "off", "consumption": 1.0, "priority": 3, "controllable": True, "priority_level": "Low"}
    ],
    "total_consumption": 0.2,  # Only refrigerator is on
    "available_power": 0.0
}

@app.on_event("startup")
async def startup_event():
    """Initialize the model and data processor on startup"""
    global predictor, data_processor
    
    print("🚀 Starting Real Solar Power Prediction API...")
    
    # Initialize data processor
    data_processor = RealSolarDataProcessor()
    
    # Initialize predictor
    predictor = RealSolarPowerPredictor()
    
    # Try to load existing model
    if os.path.exists("real_solar_power_model.pkl"):
        try:
            predictor.load_model()
            print("✅ Loaded existing trained model")
        except Exception as e:
            print(f"⚠️  Failed to load existing model: {e}")
            print("🔄 Will train new model...")
            train_model()
    else:
        print("🔄 No existing model found, training new model...")
        train_model()

def train_model():
    """Train the model with real data"""
    global predictor, data_processor
    
    try:
        print("📊 Loading real weather data...")
        weather_data = data_processor.combine_weather_data()
        
        if weather_data is None or len(weather_data) == 0:
            print("❌ No weather data available")
            return False
        
        print("🔧 Preprocessing data...")
        X, y = data_processor.preprocess_for_training()
        
        if X is None or y is None:
            print("❌ Failed to prepare training data")
            return False
        
        print("🤖 Training XGBoost model...")
        success = predictor.train_model(X, y)
        
        if success:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main dashboard"""
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_trained,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/power", response_model=PowerPredictionResponse)
async def predict_power(request: PowerPredictionRequest):
    """Predict solar power generation for given conditions"""
    if not predictor or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained or loaded")
    
    try:
        # Create feature vector
        features = {
            'hour': request.hour,
            'month': request.month,
            'hour_sin': np.sin(2 * np.pi * request.hour / 24),
            'hour_cos': np.cos(2 * np.pi * request.hour / 24),
            'month_sin': np.sin(2 * np.pi * request.month / 12),
            'month_cos': np.cos(2 * np.pi * request.month / 12),
            'ghi_normalized': request.global_irradiance / 1000,
            'temp_normalized': (request.temperature - 15) / 30,  # Normalize to 0-1
            'is_daylight': 1 if 6 <= request.hour <= 18 else 0,
            'irradiance_ratio': 0.8,  # Default clear sky ratio
            'direct_diffuse_ratio': 1.0,  # Default
            'zenith_angle': abs(90 - (request.hour - 12) * 15),
            'ghi_lag1': 0.0,  # Default
            'temp_lag1': (request.temperature - 15) / 30,
            'ghi_3h_avg': request.global_irradiance / 1000 if 6 <= request.hour <= 18 else 0.0,
            'temp_3h_avg': (request.temperature - 15) / 30
        }
        
        # Create DataFrame
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = predictor.predict(feature_df)[0]
        
        # Calculate confidence based on irradiance and time of day
        if 6 <= request.hour <= 18 and request.global_irradiance > 200:
            confidence = 0.9
        elif 6 <= request.hour <= 18:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return PowerPredictionResponse(
            predicted_power=float(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict/daily", response_model=DailyForecastResponse)
async def get_daily_forecast(
    target_date: Optional[str] = None,
    temperature: Optional[float] = None,
    irradiance: Optional[float] = None,
):
    """Get daily power forecast. If temperature/irradiance are provided,
    they will be used to build hour-wise features so results reflect inputs."""
    if not predictor or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained or loaded")
    
    try:
        # Parse target date
        if target_date:
            forecast_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        else:
            forecast_date = date.today()
        
        # If user supplied ambient conditions, build hour-wise features here
        if temperature is not None or irradiance is not None:
            month_val = (datetime.strptime(target_date, "%Y-%m-%d").month
                         if target_date else date.today().month)
            hours = list(range(24))
            rows = []
            for hr in hours:
                # Use provided values if present; otherwise fallback to simple defaults
                ghi_norm = (irradiance or (800.0 if 6 <= hr <= 18 else 0.0)) / 1000.0
                temp_norm = ((temperature if temperature is not None else 25.0) - 15.0) / 30.0
                rows.append({
                    'hour': hr,
                    'month': month_val,
                    'hour_sin': np.sin(2 * np.pi * hr / 24),
                    'hour_cos': np.cos(2 * np.pi * hr / 24),
                    'month_sin': np.sin(2 * np.pi * month_val / 12),
                    'month_cos': np.cos(2 * np.pi * month_val / 12),
                    'ghi_normalized': ghi_norm if 6 <= hr <= 18 else 0.0,
                    'temp_normalized': temp_norm,
                    'is_daylight': 1 if 6 <= hr <= 18 else 0,
                    'irradiance_ratio': 0.8,
                    'direct_diffuse_ratio': 1.0,
                    'zenith_angle': abs(90 - (hr - 12) * 15),
                    'ghi_lag1': 0.0,
                    'temp_lag1': temp_norm,
                    'ghi_3h_avg': (ghi_norm if 6 <= hr <= 18 else 0.0),
                    'temp_3h_avg': temp_norm,
                })
            feature_df = pd.DataFrame(rows)
            preds = predictor.predict(feature_df)
            forecast = [{
                'hour': h,
                'predicted_power': float(preds[i]),
                'timestamp': f"{forecast_date} {h:02d}:00:00"
            } for i, h in enumerate(hours)]
        else:
            # Fallback to model's internal simple daily forecast profile
            forecast = predictor.predict_daily_forecast(forecast_date)
        
        # Calculate statistics
        powers = [hour_data['predicted_power'] for hour_data in forecast]
        total_daily_power = sum(powers)
        peak_power = max(powers)
        peak_hour = forecast[powers.index(peak_power)]['hour']
        
        return DailyForecastResponse(
            forecast=forecast,
            total_daily_power=total_daily_power,
            peak_power=peak_power,
            peak_hour=peak_hour
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Daily forecast failed: {str(e)}")

@app.get("/digital-twin/appliances", response_model=DigitalTwinState)
async def get_appliances():
    """Get current digital twin state"""
    return DigitalTwinState(
        appliances=[ApplianceStatus(**app) for app in digital_twin_state["appliances"]],
        total_consumption=digital_twin_state["total_consumption"],
        available_power=digital_twin_state["available_power"]
    )

@app.get("/digital-twin/status", response_model=DigitalTwinState)
async def get_digital_twin_status():
    """Get summary status for the digital twin (same structure as appliances state)"""
    return DigitalTwinState(
        appliances=[ApplianceStatus(**app) for app in digital_twin_state["appliances"]],
        total_consumption=digital_twin_state["total_consumption"],
        available_power=digital_twin_state["available_power"]
    )

@app.post("/digital-twin/appliances/control")
async def control_appliance(control: ApplianceControl):
    """Control a specific appliance"""
    appliance = next((app for app in digital_twin_state["appliances"] if app["id"] == control.appliance_id), None)
    
    if not appliance:
        raise HTTPException(status_code=404, detail="Appliance not found")
    
    if not appliance["controllable"]:
        raise HTTPException(status_code=400, detail="Appliance is not controllable")
    
    # Update appliance status
    old_status = appliance["status"]
    appliance["status"] = control.action
    
    # Update total consumption
    if control.action == "on" and old_status == "off":
        digital_twin_state["total_consumption"] += appliance["consumption"]
    elif control.action == "off" and old_status == "on":
        digital_twin_state["total_consumption"] -= appliance["consumption"]
    
    return {
        "message": f"Appliance {control.appliance_id} turned {control.action}",
        "appliance_status": appliance
    }

@app.post("/digital-twin/optimize")
async def optimize_energy_usage(power_available: float = 5.0):
    """Optimize energy usage based on available solar power"""
    digital_twin_state["available_power"] = power_available
    
    # Sort appliances by priority (1=high, 2=medium, 3=low)
    controllable_apps = [
        app for app in digital_twin_state["appliances"] if app["controllable"]
    ]
    controllable_apps.sort(key=lambda x: x["priority"])
    
    current_consumption = sum(app["consumption"] for app in digital_twin_state["appliances"] if app["status"] == "on")
    
    # Turn off low-priority appliances if power_available is exceeded
    for app in controllable_apps[::-1]:  # Start with lowest priority
        if current_consumption > power_available and app["status"] == "on":
            app["status"] = "off"
            current_consumption -= app["consumption"]
            print(f"Turned off {app['name']} due to low power. Current consumption: {current_consumption:.2f} kW")
        elif current_consumption <= power_available and app["status"] == "off":
            # Try to turn on high-priority appliances if there's enough power
            if app["consumption"] <= (power_available - current_consumption):
                app["status"] = "on"
                current_consumption += app["consumption"]
                print(f"Turned on {app['name']}. Current consumption: {current_consumption:.2f} kW")
    
    digital_twin_state["total_consumption"] = current_consumption
    digital_twin_state["available_power"] = power_available
    
    return {
        "message": "Energy optimization completed",
        "available_power": power_available,
        "total_consumption": current_consumption,
        "appliances": digital_twin_state["appliances"]
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the trained model"""
    if not predictor or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained or loaded")
    
    return {
        "model_type": "XGBoost Regressor",
        "features": predictor.features,
        "feature_count": len(predictor.features),
        "trained": predictor.is_trained,
        "model_path": predictor.model_path
    }

@app.post("/train-model")
async def train_model_endpoint():
    """Trigger model training and return status"""
    success = train_model()
    if not success:
        raise HTTPException(status_code=500, detail="Model training failed")
    return {"success": True}

@app.get("/data/summary")
async def get_data_summary():
    """Get summary of loaded data"""
    if not data_processor:
        raise HTTPException(status_code=503, detail="Data processor not initialized")
    
    return data_processor.get_data_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
