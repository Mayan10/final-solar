import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealSolarDataProcessor:
    def __init__(self, data_path="."):
        self.data_path = data_path
        self.weather_data = None
        self.power_data = None
        self.combined_data = None
        
    def load_nsrdb_data(self):
        """Load NSRDB (National Solar Radiation Database) data from the large CSV files"""
        print("Loading NSRDB weather data...")
        nsrdb_files = [
            "2016-326432-one_axis.csv",
            "2017-326432-one_axis.csv", 
            "2018-326432-one_axis.csv",
            "2019-326432-one_axis.csv",
            "2020-326432-one_axis.csv"
        ]
        
        all_data = []
        for file in nsrdb_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                print(f"Processing {file}...")
                try:
                    # Try to read with different separators and handle parsing errors
                    try:
                        df = pd.read_csv(file_path, low_memory=False)
                    except pd.errors.ParserError:
                        # Try with different separator
                        df = pd.read_csv(file_path, sep='\t', low_memory=False)
                    
                    # Select relevant columns for solar power prediction
                    relevant_cols = [
                        'Year', 'Month', 'Day', 'Hour', 'Minute',
                        'Temperature', 'GHI', 'DNI', 'DHI', 
                        'Clearsky GHI', 'Clearsky DNI', 'Clearsky DHI',
                        'Solar Zenith Angle', 'Solar Azimuth Angle',
                        'Wind Speed', 'Wind Direction', 'Pressure',
                        'Relative Humidity', 'Dew Point'
                    ]
                    
                    # Filter to only include columns that exist
                    available_cols = [col for col in relevant_cols if col in df.columns]
                    if len(available_cols) < 5:  # Need at least basic columns
                        print(f"  Skipping {file} - insufficient columns")
                        continue
                        
                    df_subset = df[available_cols].copy()
                    
                    # Add year column for identification
                    df_subset['data_source'] = file
                    
                    all_data.append(df_subset)
                    print(f"  Loaded {len(df_subset)} records from {file}")
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
        
        if all_data:
            self.weather_data = pd.concat(all_data, ignore_index=True)
            print(f"Total NSRDB records loaded: {len(self.weather_data)}")
            return self.weather_data
        else:
            print("No NSRDB data loaded")
            return None
    
    def load_pv_gis_data(self):
        """Load PVGIS data from the daily data CSV files"""
        print("Loading PVGIS weather data...")
        pvgis_files = glob.glob(os.path.join(self.data_path, "Dailydata_*.csv"))
        all_data = []
        
        for file in sorted(pvgis_files):
            print(f"Processing {file}")
            try:
                # Extract month from filename
                month = int(file.split('_E5_')[1].split('_')[0])
                
                # Read the CSV file with proper handling
                with open(file, 'r') as f:
                    lines = f.readlines()
                
                # Find the data start line (after the header)
                data_start = 0
                for i, line in enumerate(lines):
                    if 'time(UTC)' in line and '\t' in line:
                        data_start = i
                        break
                
                # Read data starting from the header line
                df = pd.read_csv(file, sep='\t', skiprows=data_start, header=0)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Filter out non-data rows
                df = df[df['time(UTC)'].str.contains(r'\d{2}:\d{2}', na=False)]
                
                # Add month column
                df['month'] = month
                df['year'] = 2023  # Default year for PVGIS data
                
                # Parse time column
                df['time'] = pd.to_datetime(df['time(UTC)'], format='%H:%M', errors='coerce')
                df = df.dropna(subset=['time'])
                df['hour'] = df['time'].dt.hour
                df['minute'] = df['time'].dt.minute
                df['day'] = 1  # Default day
                
                # Convert numeric columns
                numeric_columns = ['G(i)', 'Gb(i)', 'Gd(i)', 'Gcs(i)', 'G(n)', 'Gb(n)', 'Gd(n)', 'Gcs(n)', 'T2m']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Map PVGIS columns to standard names
                df['Temperature'] = df['T2m']
                df['GHI'] = df['G(i)']
                df['DNI'] = df['Gb(i)']
                df['DHI'] = df['Gd(i)']
                df['Clearsky GHI'] = df['Gcs(i)']
                
                # Remove rows with NaN values in key columns
                df = df.dropna(subset=['GHI', 'Temperature'])
                
                if len(df) > 0:
                    all_data.append(df)
                    print(f"  Loaded {len(df)} records for month {month}")
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        if all_data:
            pvgis_data = pd.concat(all_data, ignore_index=True)
            print(f"Total PVGIS records loaded: {len(pvgis_data)}")
            return pvgis_data
        else:
            print("No PVGIS data loaded")
            return None
    
    def load_power_generation_data(self):
        """Load power generation data from Excel files"""
        print("Loading power generation data...")
        excel_files = [
            "April_13_Batch-1_Power_Generation_Data_1.xls",
            "Batch_I__n.xls", 
            "Batch_I_n.xls",
            "NVVN_B-I_Oct_13_n.xls"
        ]
        
        all_power_data = []
        for file in excel_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                print(f"Processing {file}...")
                try:
                    # Read Excel file
                    df = pd.read_excel(file_path)
                    
                    # Print structure for debugging
                    print(f"  Columns: {df.columns.tolist()}")
                    print(f"  Shape: {df.shape}")
                    
                    # Try to identify power-related columns
                    power_cols = [col for col in df.columns if any(keyword in str(col).lower() 
                                 for keyword in ['power', 'generation', 'export', 'kwh', 'mw', 'capacity'])]
                    
                    if power_cols:
                        print(f"  Power-related columns found: {power_cols}")
                        # Store the power data for potential use
                        power_df = df[power_cols].copy()
                        power_df['source_file'] = file
                        all_power_data.append(power_df)
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
        
        if all_power_data:
            self.power_data = pd.concat(all_power_data, ignore_index=True)
            print(f"Total power records loaded: {len(self.power_data)}")
            return self.power_data
        else:
            print("No power generation data loaded")
            return None
    
    def combine_weather_data(self):
        """Combine NSRDB and PVGIS weather data"""
        print("Combining weather data sources...")
        
        nsrdb_data = self.load_nsrdb_data()
        pvgis_data = self.load_pv_gis_data()
        
        combined_data = []
        
        if nsrdb_data is not None and len(nsrdb_data) > 0:
            # Standardize NSRDB data
            nsrdb_std = nsrdb_data.copy()
            nsrdb_std['data_source'] = 'NSRDB'
            combined_data.append(nsrdb_std)
            print(f"Added {len(nsrdb_std)} NSRDB records")
        
        if pvgis_data is not None and len(pvgis_data) > 0:
            # Standardize PVGIS data
            pvgis_std = pvgis_data.copy()
            pvgis_std['data_source'] = 'PVGIS'
            combined_data.append(pvgis_std)
            print(f"Added {len(pvgis_std)} PVGIS records")
        
        if combined_data:
            self.weather_data = pd.concat(combined_data, ignore_index=True)
            print(f"Total combined weather records: {len(self.weather_data)}")
            return self.weather_data
        else:
            print("No weather data available")
            return None
    
    def create_features(self, df):
        """Create features for machine learning"""
        print("Creating features...")
        
        # Ensure we have the required columns
        required_cols = ['GHI', 'Temperature', 'hour', 'month']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return None
        
        # Create time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Create solar position features
        if 'Solar Zenith Angle' in df.columns:
            df['zenith_angle'] = df['Solar Zenith Angle']
            df['zenith_angle'] = df['zenith_angle'].replace(-9999.0, np.nan)
        else:
            # Estimate zenith angle based on hour and month
            df['zenith_angle'] = 90 - (df['hour'] - 12) * 15  # Simplified calculation
            df['zenith_angle'] = np.abs(df['zenith_angle'])
        
        # Create irradiance features
        df['ghi_normalized'] = df['GHI'] / 1000  # Normalize to 0-1 range
        df['is_daylight'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
        
        # Create temperature features
        if 'Temperature' in df.columns:
            df['temp_normalized'] = (df['Temperature'] - df['Temperature'].min()) / (df['Temperature'].max() - df['Temperature'].min())
        else:
            df['temp_normalized'] = 0.5  # Default value
        
        # Create irradiance ratio features
        if 'Clearsky GHI' in df.columns:
            df['irradiance_ratio'] = df['GHI'] / (df['Clearsky GHI'] + 1e-6)
        else:
            df['irradiance_ratio'] = 0.8  # Default clear sky ratio
        
        # Create direct/diffuse ratio
        if 'DNI' in df.columns and 'DHI' in df.columns:
            df['direct_diffuse_ratio'] = df['DNI'] / (df['DHI'] + 1e-6)
        else:
            df['direct_diffuse_ratio'] = 1.0  # Default ratio
        
        # Create lagged features (simplified)
        df['ghi_lag1'] = df['GHI'].shift(1).fillna(0)
        df['temp_lag1'] = df['Temperature'].shift(1).fillna(df['Temperature'].mean())
        
        # Create rolling averages (simplified)
        df['ghi_3h_avg'] = df['GHI'].rolling(window=3, min_periods=1).mean()
        df['temp_3h_avg'] = df['Temperature'].rolling(window=3, min_periods=1).mean()
        
        # Calculate power generation potential
        # Using a more realistic model: Power = GHI * efficiency * area_factor
        efficiency = 0.2  # 20% efficiency
        area_factor = 50.0  # 50 m² panel area (typical residential system)
        df['power_potential'] = df['GHI'] * efficiency * area_factor / 1000  # Convert to kW
        
        print(f"Created features for {len(df)} records")
        return df
    
    def preprocess_for_training(self, df=None):
        """Preprocess data for machine learning training"""
        if df is None:
            df = self.weather_data
        
        if df is None or len(df) == 0:
            print("No data available for preprocessing")
            return None, None
        
        print("Preprocessing data for training...")
        
        # Create features
        df = self.create_features(df)
        
        if df is None:
            return None, None
        
        # Select features for training
        feature_columns = [
            'hour', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'ghi_normalized', 'temp_normalized', 'is_daylight', 'irradiance_ratio',
            'direct_diffuse_ratio', 'zenith_angle', 'ghi_lag1', 'temp_lag1',
            'ghi_3h_avg', 'temp_3h_avg'
        ]
        
        # Filter to only include available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            print("No features available for training")
            return None, None
        
        # Prepare features and target
        X = df[available_features].fillna(0)
        y = df['power_potential'].fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Features used: {available_features}")
        
        return X, y
    
    def get_data_summary(self):
        """Get summary of loaded data"""
        summary = {
            'weather_data_loaded': self.weather_data is not None,
            'power_data_loaded': self.power_data is not None,
            'weather_records': len(self.weather_data) if self.weather_data is not None else 0,
            'power_records': len(self.power_data) if self.power_data is not None else 0
        }
        
        if self.weather_data is not None:
            summary['weather_columns'] = list(self.weather_data.columns)
            summary['weather_date_range'] = {
                'min_year': self.weather_data.get('year', pd.Series([0])).min(),
                'max_year': self.weather_data.get('year', pd.Series([0])).max(),
                'months_available': sorted(self.weather_data.get('month', pd.Series([0])).unique())
            }
        
        return summary

def main():
    """Main function to test the data processor"""
    processor = RealSolarDataProcessor()
    
    # Load and combine weather data
    weather_data = processor.combine_weather_data()
    
    if weather_data is not None:
        print(f"\nWeather data summary:")
        print(f"Records: {len(weather_data)}")
        print(f"Columns: {list(weather_data.columns)}")
        
        # Create features and prepare for training
        X, y = processor.preprocess_for_training()
        
        if X is not None and y is not None:
            print(f"\nTraining data prepared:")
            print(f"Features: {X.shape}")
            print(f"Target: {y.shape}")
            print(f"Feature names: {list(X.columns)}")
            
            # Show some statistics
            print(f"\nTarget statistics:")
            print(f"Mean power: {y.mean():.2f} kW")
            print(f"Max power: {y.max():.2f} kW")
            print(f"Min power: {y.min():.2f} kW")
        else:
            print("Failed to prepare training data")
    else:
        print("No weather data loaded")

if __name__ == "__main__":
    main()
