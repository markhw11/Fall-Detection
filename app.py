from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import tensorflow as tf
import pandas as pd
import uvicorn
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fall Detection API - Pure ML Predictions", 
    description="API for fall detection relying solely on machine learning model predictions",
    version="2.1.0"
)

# Global model variable
model = None

# Load the enhanced anti-overfitting trained model
def load_model(model_path: str = "enhanced_anti_overfitting_fall_detection_model.h5"):
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        return False

# Load model on startup
load_model()

# Pydantic models for request validation
class SensorReading(BaseModel):
    time: Optional[float] = None
    ax: float  # Accelerometer X
    ay: float  # Accelerometer Y
    az: float  # Accelerometer Z
    wx: float  # Gyroscope X
    wy: float  # Gyroscope Y
    wz: float  # Gyroscope Z
    
    @validator('ax', 'ay', 'az', 'wx', 'wy', 'wz')
    def validate_sensor_values(cls, v):
        if not -100 <= v <= 100:  # Reasonable sensor value range
            raise ValueError('Sensor values must be between -100 and 100')
        return v

class FallDetectionData(BaseModel):
    sensor_data: List[SensorReading]
    
    @validator('sensor_data')
    def validate_sensor_data_length(cls, v):
        if len(v) != 100:
            raise ValueError('sensor_data must contain exactly 100 readings')
        return v

class FallDetectionRawData(BaseModel):
    features: List[List[float]]
    
    @validator('features')
    def validate_features_shape(cls, v):
        if len(v) != 100:
            raise ValueError('features must contain exactly 100 time steps')
        if any(len(timestep) != 27 for timestep in v):
            raise ValueError('Each timestep must have exactly 27 features')
        return v

# List of class labels (must match training order)
classes = ['falling', 'kneeling', 'walking']

def calculate_change_features(window_data):
    """
    Calculate change detection features for a window of sensor data.
    Returns 15 features to combine with 12 base features for 27 total.
    """
    features = []
    
    # Use the already computed acc_mag and gyro_mag if available
    if 'acc_mag' in window_data.columns:
        acc_mag = window_data['acc_mag']
    else:
        acc_mag = np.sqrt(window_data['ax']**2 + window_data['ay']**2 + window_data['az']**2)
    
    if 'gyro_mag' in window_data.columns:
        gyro_mag = window_data['gyro_mag']
    else:
        gyro_mag = np.sqrt(window_data['wx']**2 + window_data['wy']**2 + window_data['wz']**2)
    
    # 1. Maximum acceleration magnitude
    max_acc = acc_mag.max()
    features.append(max_acc)
    
    # 2. Maximum change in acceleration magnitude
    acc_diff = acc_mag.diff().abs().max()
    features.append(acc_diff if not np.isnan(acc_diff) else 0)
    
    # 3. Standard deviation of acceleration
    acc_std = acc_mag.std()
    features.append(acc_std if not np.isnan(acc_std) else 0)
    
    # 4. Maximum rotational velocity (using gyro_mag)
    max_gyro = gyro_mag.max()
    features.append(max_gyro)
    
    # 5. Free fall indicator (minimum acceleration)
    min_acc = acc_mag.min()
    features.append(min_acc)
    
    # 6. Impact indicator
    acc_values = acc_mag.values
    impact_score = 0
    for i in range(1, len(acc_values)):
        if acc_values[i-1] < 5.0 and acc_values[i] > 12.0:
            impact_score = max(impact_score, acc_values[i] - acc_values[i-1])
    features.append(impact_score)
    
    # 7. Change rate in X, Y, Z accelerations
    ax_change = window_data['ax'].diff().abs().max()
    ay_change = window_data['ay'].diff().abs().max()
    az_change = window_data['az'].diff().abs().max()
    features.extend([
        ax_change if not np.isnan(ax_change) else 0,
        ay_change if not np.isnan(ay_change) else 0,
        az_change if not np.isnan(az_change) else 0
    ])
    
    # 8. Gyroscope change features
    wx_change = window_data['wx'].diff().abs().max()
    wy_change = window_data['wy'].diff().abs().max()
    wz_change = window_data['wz'].diff().abs().max()
    features.extend([
        wx_change if not np.isnan(wx_change) else 0,
        wy_change if not np.isnan(wy_change) else 0,
        wz_change if not np.isnan(wz_change) else 0
    ])
    
    # 9. Statistical features from rolling windows
    if 'acc_mag_std_50' in window_data.columns:
        acc_std_50_max = window_data['acc_mag_std_50'].max()
        gyro_std_50_max = window_data['gyro_mag_std_50'].max()
        features.extend([
            acc_std_50_max if not np.isnan(acc_std_50_max) else 0,
            gyro_std_50_max if not np.isnan(gyro_std_50_max) else 0
        ])
    else:
        features.extend([0, 0])  # Placeholder if rolling features not available
    
    # 10. Fall pattern score (kept for feature consistency but not used in decision)
    fall_score = 0
    if max_acc > 15.0:
        fall_score += 0.3
    if min_acc < 3.0:
        fall_score += 0.25
    if acc_diff > 8.0:
        fall_score += 0.25
    if max_gyro > 3.0:
        fall_score += 0.2
    if len(features) >= 12 and features[12] > 2.0:
        fall_score += 0.1
    if len(features) >= 13 and features[13] > 1.0:
        fall_score += 0.1
    
    features.append(fall_score)
    
    return np.array(features).reshape(1, -1)

def preprocess_enhanced_sensor_data(sensor_readings: List[SensorReading]) -> np.ndarray:
    """
    Preprocess sensor data with enhanced features matching the training pipeline.
    """
    try:
        # Convert to DataFrame
        data = []
        for reading in sensor_readings:
            data.append([
                reading.ax, reading.ay, reading.az,
                reading.wx, reading.wy, reading.wz
            ])
        
        df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'wx', 'wy', 'wz'])
        
        # Compute magnitude features
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['gyro_mag'] = np.sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)
        
        # Rolling window statistics (50-sample windows)
        df['acc_mag_std_50'] = df['acc_mag'].rolling(window=50, min_periods=1).std().fillna(0)
        df['gyro_mag_std_50'] = df['gyro_mag'].rolling(window=50, min_periods=1).std().fillna(0)
        df['acc_mag_mean_50'] = df['acc_mag'].rolling(window=50, min_periods=1).mean().fillna(0)
        df['gyro_mag_mean_50'] = df['gyro_mag'].rolling(window=50, min_periods=1).mean().fillna(0)
        
        # Base features (must match training feature_columns)
        feature_columns = [
            'ax', 'ay', 'az', 'wx', 'wy', 'wz',
            'acc_mag', 'gyro_mag',
            'acc_mag_std_50', 'gyro_mag_std_50',
            'acc_mag_mean_50', 'gyro_mag_mean_50'
        ]
        
        base_features = df[feature_columns].values
        
        # Calculate change detection features
        change_features = calculate_change_features(df)
        
        # Create enhanced features (base + change features repeated for each timestep)
        enhanced_features = np.concatenate([
            base_features, 
            np.tile(change_features, (len(df), 1))
        ], axis=1)
        
        return enhanced_features
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/predict/")
def predict(data: FallDetectionData):
    """
    Fall detection using pure machine learning model predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess with enhanced features
        features = preprocess_enhanced_sensor_data(data.sensor_data)
        
        # Validate feature shape - model expects 27 features (12 base + 15 change)
        expected_features = 27
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid feature shape after preprocessing. Expected (100, {expected_features}), got {features.shape}"
            )
        
        # Reshape for prediction
        features = np.expand_dims(features, axis=0)
        
        # Get model prediction
        prediction = model.predict(features, verbose=0)
        
        # Extract probabilities
        falling_prob = float(prediction[0][0])
        kneeling_prob = float(prediction[0][1])
        walking_prob = float(prediction[0][2])
        
        # Use pure ML predictions - no rule-based overrides
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Get prediction probabilities for all classes
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "decision_reason": f"ML model prediction: {predicted_class}",
            "ml_predictions": prediction_probs,
            "model_info": {
                "model_type": "Pure ML Fall Detection",
                "features_used": expected_features,
                "feature_breakdown": "12 base + 15 change detection = 27 total",
                "decision_approach": "Pure machine learning - no rule-based overrides"
            },
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_raw/")
def predict_raw(data: FallDetectionRawData):
    """
    Alternative endpoint for already preprocessed data.
    Expects 100 time steps with 27 features each.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert the input list into a numpy array
        features = np.array(data.features)
        
        # Validate input shape (100 time steps, 27 features)
        expected_features = 27
        if features.shape != (100, expected_features):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape. Expected (100, {expected_features}), got {features.shape}"
            )
        
        # Reshape input to match the model's expected shape
        features = np.expand_dims(features, axis=0)
        
        # Get model prediction
        prediction = model.predict(features, verbose=0)
        
        # Use pure ML predictions
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "decision_reason": f"ML model prediction: {predicted_class} (raw data mode)",
            "ml_predictions": prediction_probs,
            "model_info": {
                "model_type": "Pure ML Fall Detection",
                "decision_approach": "Pure machine learning - no rule-based overrides"
            },
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Raw prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Raw prediction failed: {str(e)}")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Pure ML Fall Detection API!",
        "model_loaded": model is not None,
        "model_info": {
            "type": "Pure Machine Learning Fall Detection",
            "approach": "Relies solely on trained model predictions", 
            "features_per_timestep": 27,
            "feature_breakdown": "12 base + 15 change detection = 27 total"
        },
        "features": [
            "✓ Pure machine learning approach",
            "✓ Bidirectional GRU neural network",
            "✓ Enhanced feature preprocessing",
            "✓ Rolling window statistical features",
            "✓ Change detection features for context",
            "✓ No rule-based overrides",
            "✓ Clean ML-based decision making"
        ],
        "classes": classes,
        "endpoints": {
            "/predict/": "Send 100 sensor readings with automatic feature enhancement",
            "/predict_raw/": "Send preprocessed data (27 features per timestep)",
            "/health": "Check API and model status",
            "/model_info": "Get detailed model information",
            "/reload_model": "Reload the model (POST endpoint)"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "Pure ML Fall Detection",
        "approach": "Machine learning predictions only"
    }

@app.post("/reload_model")
def reload_model(model_path: str = "enhanced_anti_overfitting_fall_detection_model.h5"):
    """Reload the model from file"""
    success = load_model(model_path)
    if success:
        return {"status": "success", "message": f"Model reloaded from {model_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/model_info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "classes": classes,
            "features": {
                "base_features": [
                    "ax", "ay", "az", "wx", "wy", "wz",
                    "acc_mag", "gyro_mag",
                    "acc_mag_std_50", "gyro_mag_std_50",
                    "acc_mag_mean_50", "gyro_mag_mean_50"
                ],
                "change_detection_features": [
                    "max_acceleration", "max_change_rate", "acceleration_std",
                    "max_gyro_velocity", "min_acceleration", "impact_score",
                    "ax_change", "ay_change", "az_change",
                    "wx_change", "wy_change", "wz_change",
                    "acc_std_50_max", "gyro_std_50_max",
                    "fall_pattern_score"
                ],
                "total_features_per_timestep": 27
            },
            "model_approach": {
                "decision_method": "Pure machine learning",
                "rule_based_overrides": "None - removed",
                "fall_indicators": "Not used in decision making",
                "confidence_source": "Direct from neural network output"
            },
            "preprocessing": {
                "feature_engineering": "✓ Enhanced features with rolling statistics",
                "change_detection": "✓ 15 change detection features (for context only)",
                "normalization": "✓ Handled during training"
            }
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Pure ML Fall Detection API...")
    print("Model features:")
    print("- ✓ Pure machine learning approach")
    print("- ✓ No rule-based overrides or fall indicators")
    print("- ✓ Enhanced feature preprocessing")
    print("- ✓ Direct neural network predictions")
    print("- ✓ Clean decision making process")
    uvicorn.run(app, host="0.0.0.0", port=8000)
