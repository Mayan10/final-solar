# Solar Power Prediction & Digital Twin System

An AI-powered system for predicting solar power generation and controlling smart appliances through a digital twin interface.

## Features

- **Solar Power Prediction**: XGBoost-based ML model for accurate solar power forecasting
- **Digital Twin**: Virtual representation of home appliances with smart control
- **Real-time Dashboard**: Interactive web interface for monitoring and control
- **Energy Optimization**: Automatic appliance management based on available solar power
- **RESTful API**: Complete API for integration with other systems

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Weather Data  │    │   Power Data    │    │   ML Model      │
│   (CSV Files)   │───▶│   (Excel Files) │───▶│   (XGBoost)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │◀───│   Data          │───▶│   Predictions   │
│   Backend       │    │   Processor     │    │   & Analytics   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │◀───│   Digital Twin  │
│   (HTML/JS)     │    │   Controller    │
└─────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd XGBR_SOLAR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**:
   ```bash
   python run.py
   ```

## Project Structure

```
XGBR_SOLAR/
├── api.py                 # FastAPI backend
├── data_processor.py      # Data loading and preprocessing
├── model_trainer.py       # XGBoost model training
├── run.py                # Main entry point
├── requirements.txt      # Python dependencies
├── static/
│   └── index.html        # Web frontend
├── data/                 # Raw data files
└── models/              # Trained models
```

## API Endpoints

### Power Prediction
- `POST /predict/power` - Predict power for specific conditions
- `POST /predict/daily` - Predict power for entire day
- `POST /train-model` - Train/retrain the model

### Digital Twin
- `GET /digital-twin/status` - Get current system status
- `GET /digital-twin/appliances` - List all appliances
- `POST /digital-twin/control` - Control specific appliance
- `POST /digital-twin/optimize` - Optimize energy usage

### System
- `GET /` - Web dashboard
- `GET /health` - Health check
- `GET /docs` - API documentation

## Usage Examples

### 1. Predict Solar Power
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict/power", json={
    "hour": 12,
    "temperature": 25.5,
    "global_irradiance": 800,
    "direct_irradiance": 600,
    "diffuse_irradiance": 200,
    "month": 6
})

print(f"Predicted Power: {response.json()['power_kw']} kW")
```

### 2. Control Digital Twin
```python
# Turn on air conditioner
response = requests.post("http://localhost:8000/digital-twin/control", json={
    "appliance_id": "ac_1",
    "action": "on"
})

# Optimize energy usage
response = requests.post("http://localhost:8000/digital-twin/optimize", json=5.0)  # 5kW available
```

### 3. Daily Prediction
```python
# Predict full day
weather_forecast = [
    {"hour": h, "temperature": 25, "global_irradiance": 800, 
     "direct_irradiance": 600, "diffuse_irradiance": 200, "month": 6}
    for h in range(24)
]

response = requests.post("http://localhost:8000/predict/daily", json={
    "prediction_date": "2024-06-15",
    "weather_forecast": weather_forecast
})

daily_data = response.json()
print(f"Total Power: {daily_data['total_power_kwh']} kWh")
```

## Digital Twin Appliances

The system includes these controllable appliances:

| Appliance | Power (kW) | Priority | Controllable |
|-----------|------------|----------|--------------|
| Air Conditioner 1 | 2.0 | High | Yes |
| Air Conditioner 2 | 2.0 | Medium | Yes |
| Water Heater | 1.5 | Medium | Yes |
| Washing Machine | 1.0 | Medium | Yes |
| TV | 0.2 | Low | Yes |
| LED Lights | 0.1 | Low | Yes |
| Refrigerator | 0.2 | High | No |

## Model Performance

The XGBoost model uses these features:
- **Time Features**: Hour, month (cyclical encoding)
- **Weather Features**: Temperature, irradiance, humidity
- **Solar Features**: Direct/diffuse irradiance ratios
- **Historical Features**: Rolling averages, lag features

Typical performance metrics:
- **R² Score**: 0.85-0.95
- **RMSE**: 50-150 W
- **MAE**: 30-100 W

## Energy Optimization

The system automatically optimizes energy usage by:
1. **Priority-based Control**: High-priority appliances get power first
2. **Load Balancing**: Distribute available power efficiently
3. **Smart Scheduling**: Turn off non-essential appliances when power is low
4. **Real-time Monitoring**: Continuous power consumption tracking

## Web Interface

Access the dashboard at `http://localhost:8000` to:
- View real-time power predictions
- Control appliances through the digital twin
- Monitor energy consumption
- Optimize energy usage
- View historical data and trends

## Configuration

### Model Parameters
```python
# XGBoost parameters (in model_trainer.py)
n_estimators=100
max_depth=6
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
```

### Digital Twin Settings
```python
# Appliance priorities (in api.py)
# 1 = High priority (essential)
# 2 = Medium priority (important)
# 3 = Low priority (optional)
```

## Troubleshooting

### Common Issues

1. **Model not trained**:
   - Run `POST /train-model` endpoint
   - Check data files are present

2. **API connection failed**:
   - Ensure server is running on port 8000
   - Check firewall settings

3. **Prediction errors**:
   - Verify input data format
   - Check model is loaded

### Logs
- Server logs: Console output
- Model logs: `model_analysis.png`
- Error logs: API response messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PVGIS data from European Union
- XGBoost library for machine learning
- FastAPI for the web framework
- Plotly for visualizations

---
