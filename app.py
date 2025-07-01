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
def load_model(model_path: str = "anti_overfitting_fall_detection_model.h5"):
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
        if any(len(timestep) != 17 for timestep in v):
            raise ValueError('Each timestep must have exactly 17 features')
        return v

# List of class labels (must match training order)
classes = ['falling', 'kneeling', 'walking']

# Removed change detection features - model now expects only 17 base features

def preprocess_sensor_data(sensor_readings: List[SensorReading]) -> np.ndarray:
    """
    Preprocess sensor data with 17 base features for the simplified model.
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
        
        # Additional engineered features for 17 total
        df['acc_x_mag'] = df['ax'].abs()
        df['acc_y_mag'] = df['ay'].abs()
        df['acc_z_mag'] = df['az'].abs()
        df['gyro_x_mag'] = df['wx'].abs()
        df['gyro_y_mag'] = df['wy'].abs()
        
        # 17 features total (must match model training)
        feature_columns = [
            'ax', 'ay', 'az', 'wx', 'wy', 'wz',           # 6 raw features
            'acc_mag', 'gyro_mag',                          # 2 magnitude features  
            'acc_mag_std_50', 'gyro_mag_std_50',           # 2 rolling std features
            'acc_mag_mean_50', 'gyro_mag_mean_50',         # 2 rolling mean features
            'acc_x_mag', 'acc_y_mag', 'acc_z_mag',         # 3 absolute acceleration features
            'gyro_x_mag', 'gyro_y_mag'                     # 2 absolute gyro features
        ]
        
        features = df[feature_columns].values
        
        return features
        
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
        # Preprocess with base features only
        features = preprocess_sensor_data(data.sensor_data)
        
        # Validate feature shape - model expects 17 features
        expected_features = 17
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
                "feature_breakdown": "17 engineered features",
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
    Expects 100 time steps with 17 features each.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert the input list into a numpy array
        features = np.array(data.features)
        
        # Validate input shape (100 time steps, 17 features)
        expected_features = 17
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
            "features_per_timestep": 17,
            "feature_breakdown": "17 engineered features"
        },
        "features": [
            "✓ Pure machine learning approach",
            "✓ Bidirectional GRU neural network",
            "✓ 17 engineered features",
            "✓ Rolling window statistical features",
            "✓ Magnitude and absolute value features",
            "✓ No rule-based overrides",
            "✓ Clean ML-based decision making"
        ],
        "classes": classes,
        "endpoints": {
            "/predict/": "Send 100 sensor readings with automatic feature enhancement",
            "/predict_raw/": "Send preprocessed data (17 features per timestep)",
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
                    "acc_mag_mean_50", "gyro_mag_mean_50",
                    "acc_x_mag", "acc_y_mag", "acc_z_mag",
                    "gyro_x_mag", "gyro_y_mag"
                ],
                "total_features_per_timestep": 17
            },
            "model_approach": {
                "decision_method": "Pure machine learning",
                "rule_based_overrides": "None - removed",
                "fall_indicators": "Not used in decision making",
                "confidence_source": "Direct from neural network output"
            },
            "preprocessing": {
                "feature_engineering": "✓ 17 engineered features with rolling statistics",
                "magnitude_features": "✓ Acceleration and gyroscope magnitudes",
                "absolute_features": "✓ Absolute values for directional components",
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
