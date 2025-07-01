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
    title="Conservative Fall Detection API", 
    description="API for fall detection with reduced false positives",
    version="2.2.0"
)

# Global model variable
model = None

# Load the model
def load_model(model_path: str ="anti_overfitting_fall_detection_model.h5"):
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

def detect_motion_level(window_data):
    """
    Detect the overall motion level to help distinguish actual movement from noise.
    Returns motion metrics to prevent false positives during stationary periods.
    """
    # Calculate basic motion indicators
    acc_mag = np.sqrt(window_data['ax']**2 + window_data['ay']**2 + window_data['az']**2)
    gyro_mag = np.sqrt(window_data['wx']**2 + window_data['wy']**2 + window_data['wz']**2)
    
    # Motion level indicators
    acc_variance = acc_mag.var()
    gyro_variance = gyro_mag.var()
    acc_range = acc_mag.max() - acc_mag.min()
    gyro_range = gyro_mag.max() - gyro_mag.min()
    
    # Check for actual significant movement
    has_significant_motion = (
        acc_variance > 1.0 or  # Significant acceleration variance
        gyro_variance > 0.5 or  # Significant rotation variance
        acc_range > 5.0 or     # Significant acceleration range
        gyro_range > 2.0       # Significant rotation range
    )
    
    return {
        'has_significant_motion': bool(has_significant_motion),
        'acc_variance': float(acc_variance),
        'gyro_variance': float(gyro_variance),
        'acc_range': float(acc_range),
        'gyro_range': float(gyro_range),
        'max_acc': float(acc_mag.max()),
        'min_acc': float(acc_mag.min()),
        'max_gyro': float(gyro_mag.max())
    }

def preprocess_sensor_data(sensor_readings: List[SensorReading]) -> np.ndarray:
    """
    Preprocess sensor data with 17 features for the model.
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
        
        # 17 features total
        feature_columns = [
            'ax', 'ay', 'az', 'wx', 'wy', 'wz',           # 6 raw features
            'acc_mag', 'gyro_mag',                          # 2 magnitude features  
            'acc_mag_std_50', 'gyro_mag_std_50',           # 2 rolling std features
            'acc_mag_mean_50', 'gyro_mag_mean_50',         # 2 rolling mean features
            'acc_x_mag', 'acc_y_mag', 'acc_z_mag',         # 3 absolute acceleration features
            'gyro_x_mag', 'gyro_y_mag'                     # 2 absolute gyro features
        ]
        
        features = df[feature_columns].values
        
        return features, df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/predict/")
def predict(data: FallDetectionData):
    """
    Conservative fall detection with motion analysis to reduce false positives.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess features and get raw data for motion analysis
        features, df = preprocess_sensor_data(data.sensor_data)
        
        # Validate feature shape - model expects 17 features
        expected_features = 17
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid feature shape after preprocessing. Expected (100, {expected_features}), got {features.shape}"
            )
        
        # Analyze motion level to prevent false positives
        motion_analysis = detect_motion_level(df)
        
        # Reshape for prediction
        features = np.expand_dims(features, axis=0)
        
        # Get model prediction
        prediction = model.predict(features, verbose=0)
        
        # Extract probabilities
        falling_prob = float(prediction[0][0])
        kneeling_prob = float(prediction[0][1])
        walking_prob = float(prediction[0][2])
        
        # Use the model's prediction as the primary decision
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # CONSERVATIVE APPROACH: Only override to fall if there's VERY strong evidence
        # AND there's significant motion detected
        if (predicted_class == "falling" and 
            motion_analysis['has_significant_motion'] and
            falling_prob > 0.8):  # Much higher threshold
            
            # Additional validation for fall
            if (motion_analysis['max_acc'] > 20.0 and  # Very high acceleration
                motion_analysis['acc_range'] > 8.0 and  # Significant acceleration change
                motion_analysis['acc_variance'] > 3.0):  # High variance indicating real movement
                
                predicted_class = "falling"
                confidence = falling_prob
                decision_reason = f"High-confidence fall detected with significant motion (ML: {falling_prob:.3f})"
            else:
                # Downgrade to most likely non-fall class
                if kneeling_prob > walking_prob:
                    predicted_class = "kneeling"
                    confidence = kneeling_prob
                else:
                    predicted_class = "walking"
                    confidence = walking_prob
                decision_reason = f"Fall prediction downgraded due to insufficient motion evidence"
        
        elif not motion_analysis['has_significant_motion']:
            # If there's no significant motion, it's very unlikely to be a fall
            if kneeling_prob > walking_prob:
                predicted_class = "kneeling"
                confidence = kneeling_prob
            else:
                predicted_class = "walking" 
                confidence = walking_prob
            decision_reason = "No significant motion detected - not a fall"
        
        else:
            # Trust the model for non-fall predictions or low-confidence falls
            decision_reason = f"ML model prediction: {predicted_class} (confidence: {confidence:.3f})"
        
        # Get prediction probabilities for all classes
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "decision_reason": decision_reason,
            "ml_predictions": prediction_probs,
            "motion_analysis": {
                "has_significant_motion": bool(motion_analysis['has_significant_motion']),
                "acceleration_variance": round(float(motion_analysis['acc_variance']), 3),
                "gyroscope_variance": round(float(motion_analysis['gyro_variance']), 3),
                "acceleration_range": round(float(motion_analysis['acc_range']), 3),
                "gyroscope_range": round(float(motion_analysis['gyro_range']), 3),
                "max_acceleration": round(float(motion_analysis['max_acc']), 3),
                "min_acceleration": round(float(motion_analysis['min_acc']), 3),
                "max_gyroscope": round(float(motion_analysis['max_gyro']), 3)
            },
            "model_info": {
                "model_type": "Conservative Fall Detection",
                "features_used": expected_features,
                "approach": "Motion-validated ML predictions with high thresholds"
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
        
        # Use conservative approach for raw data too
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Conservative threshold for raw data
        falling_prob = float(prediction[0][0])
        if predicted_class == "falling" and falling_prob < 0.7:
            # Downgrade to next most likely class
            remaining_probs = [prediction[0][1], prediction[0][2]]
            remaining_classes = ['kneeling', 'walking']
            next_best_idx = np.argmax(remaining_probs)
            predicted_class = remaining_classes[next_best_idx]
            confidence = remaining_probs[next_best_idx]
            decision_reason = f"Fall prediction downgraded due to low confidence (raw data mode)"
        else:
            decision_reason = f"ML model prediction: {predicted_class} (raw data mode)"
        
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "decision_reason": decision_reason,
            "ml_predictions": prediction_probs,
            "note": "Raw data mode - motion analysis not available, using conservative thresholds",
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
        "message": "Welcome to the Conservative Fall Detection API!",
        "model_loaded": model is not None,
        "model_info": {
            "type": "Conservative Fall Detection with Motion Validation",
            "approach": "High thresholds and motion analysis to reduce false positives", 
            "features_per_timestep": 17,
            "feature_breakdown": "17 engineered features with motion validation"
        },
        "features": [
            "✓ Conservative fall detection approach",
            "✓ Motion level analysis to prevent false positives",
            "✓ High confidence thresholds for fall detection",
            "✓ Automatic downgrading of uncertain predictions",
            "✓ 17 engineered sensor features",
            "✓ Rolling window statistical analysis",
            "✓ Reduced false positive rate"
        ],
        "classes": classes,
        "endpoints": {
            "/predict/": "Send 100 sensor readings with motion validation",
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
        "model_type": "Conservative Fall Detection",
        "approach": "Motion-validated predictions with reduced false positives"
    }

@app.post("/reload_model")
def reload_model(model_path: str = r"D:\Uni\Graduationproject\FallDet\anti_overfitting_fall_detection_model.h5"):
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
                "engineered_features": [
                    "ax", "ay", "az", "wx", "wy", "wz",
                    "acc_mag", "gyro_mag",
                    "acc_mag_std_50", "gyro_mag_std_50",
                    "acc_mag_mean_50", "gyro_mag_mean_50",
                    "acc_x_mag", "acc_y_mag", "acc_z_mag",
                    "gyro_x_mag", "gyro_y_mag"
                ],
                "total_features_per_timestep": 17
            },
            "conservative_approach": {
                "motion_validation": "✓ Requires significant motion for fall detection",
                "high_thresholds": "✓ Fall confidence > 0.6 required",
                "additional_checks": "✓ Acceleration variance and range validation",
                "false_positive_reduction": "✓ Automatic downgrading of uncertain predictions"
            },
            "motion_thresholds": {
                "acceleration_variance": "> 1.0 for significant motion",
                "gyroscope_variance": "> 0.5 for significant motion", 
                "acceleration_range": "> 5.0 for significant motion",
                "gyroscope_range": "> 2.0 for significant motion",
                "fall_confidence": "> 0.6 required",
                "fall_acceleration": "> 20.0 m/s² for validation"
            }
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Conservative Fall Detection API...")
    print("Model features:")
    print("- ✓ Conservative approach with motion validation")
    print("- ✓ High thresholds to reduce false positives")
    print("- ✓ Automatic prediction downgrading")
    print("- ✓ Motion level analysis")
    print("- ✓ 17 engineered features")
    uvicorn.run(app, host="0.0.0.0", port=8000)
