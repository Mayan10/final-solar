#!/usr/bin/env python3
"""
Real Solar Power Prediction System
Trains model with actual data and starts the API server
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def cleanup_demo_files():
    """Remove unnecessary demo and test files"""
    print("🧹 Cleaning up demo files...")
    
    files_to_remove = [
        "demo.py",
        "test_system.py", 
        "synthetic_data_generator.py",
        "data_processor.py",  # Old version
        "model_trainer.py",   # Old version
        "api.py",             # Old version
        "run.py",             # Old version
        "SYSTEM_SUMMARY.md",
        "model_analysis.png",  # Old analysis
        "solar_power_model.pkl"  # Old model
    ]
    
    removed_count = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  ✅ Removed {file}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not remove {file}: {e}")
    
    print(f"🧹 Cleaned up {removed_count} files")

def train_real_model():
    """Train the model with real data"""
    print("🤖 Training model with real data...")
    
    try:
        from real_model_trainer import main as train_main
        train_main()
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def start_api_server():
    """Start the API server"""
    print("🚀 Starting API server...")
    
    try:
        import uvicorn
        from real_api import app
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")

def main():
    """Main function"""
    print("="*60)
    print("🌞 REAL SOLAR POWER PREDICTION SYSTEM")
    print("="*60)
    print("Using actual weather data and power generation data")
    print("="*60)
    
    # Step 1: Clean up demo files
    cleanup_demo_files()
    
    # Step 2: Train the real model
    print("\n📊 Step 1: Training model with real data...")
    if not train_real_model():
        print("❌ Model training failed. Exiting.")
        return
    
    print("✅ Model training completed successfully!")
    
    # Step 3: Start API server
    print("\n🚀 Step 2: Starting API server...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
