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
    title="Enhanced Anti-Overfitting Fall Detection API", 
    description="API for fall detection with change detection and anti-overfitting",
    version="2.0.0"
)

# Global model variable
model = None

# Load the enhanced anti-overfitting trained model
def load_model(model_path: str = "enhanced_anti_overfitting_fall_detection_model.h5"):
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Enhanced anti-overfitting model loaded successfully from {model_path}")
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
        if any(len(timestep) != 17 for timestep in v):  # Updated for correct features
            raise ValueError('Each timestep must have exactly 17 features')
        return v

# List of class labels (must match training order)
classes = ['falling', 'kneeling', 'walking']

def calculate_change_features(window_data):
    """
    Calculate drastic change indicators for a window of sensor data.
    FIXED to match training exactly - produces 15 features for 27 total!
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
    
    # Model expects 27 total features = 12 base + 15 change features
    
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
    
    # 9. Statistical features from rolling windows (if available)
    if 'acc_mag_std_50' in window_data.columns:
        acc_std_50_max = window_data['acc_mag_std_50'].max()
        gyro_std_50_max = window_data['gyro_mag_std_50'].max()
        features.extend([
            acc_std_50_max if not np.isnan(acc_std_50_max) else 0,
            gyro_std_50_max if not np.isnan(gyro_std_50_max) else 0
        ])
    else:
        features.extend([0, 0])  # Placeholder if rolling features not available
    
    # 10. Enhanced fall pattern score
    fall_score = 0
    if max_acc > 15.0:
        fall_score += 0.3
    if min_acc < 3.0:
        fall_score += 0.25
    if acc_diff > 8.0:
        fall_score += 0.25
    if max_gyro > 3.0:
        fall_score += 0.2
    # Add statistical indicators
    if len(features) >= 14 and features[12] > 2.0:  # High acc std
        fall_score += 0.1
    if len(features) >= 14 and features[13] > 1.0:  # High gyro std
        fall_score += 0.1
    
    features.append(fall_score)
    
    return np.array(features).reshape(1, -1)

def preprocess_enhanced_sensor_data(sensor_readings: List[SensorReading]) -> tuple:
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
        
        return enhanced_features, change_features
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/predict/")
def predict(data: FallDetectionData):
    """
    Enhanced fall detection with anti-overfitting model and change pattern analysis.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess with enhanced features
        features, change_features = preprocess_enhanced_sensor_data(data.sensor_data)
        
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
        
        # Extract change analysis features (15 core features)
        max_acc = float(change_features[0][0])
        max_change = float(change_features[0][1])
        acc_std = float(change_features[0][2])
        max_gyro = float(change_features[0][3])
        min_acc = float(change_features[0][4])
        impact_score = float(change_features[0][5])
        ax_change = float(change_features[0][6])
        ay_change = float(change_features[0][7])
        az_change = float(change_features[0][8])
        wx_change = float(change_features[0][9])
        wy_change = float(change_features[0][10])
        wz_change = float(change_features[0][11])
        acc_std_50_max = float(change_features[0][12])
        gyro_std_50_max = float(change_features[0][13])
        fall_score = float(change_features[0][14])
        
        # Enhanced decision logic (matching training approach)
        predicted_class_index = prediction.argmax(axis=1)[0]
        ml_top_choice = classes[predicted_class_index]
        ml_confidence = float(prediction[0][predicted_class_index])
        
        # Apply enhanced decision rules
        if fall_score > 0.9 and max_acc > 30.0:  # Extreme fall pattern
            predicted_class = "falling"
            confidence = min(0.95, falling_prob + fall_score * 0.5)
            decision_reason = "OVERRIDE: Extreme fall pattern detected"
        elif max_acc > 35.0 and min_acc < 1.0:  # Massive impact + free fall
            predicted_class = "falling"
            confidence = min(0.90, falling_prob + 0.4)
            decision_reason = "OVERRIDE: Massive impact + free fall"
        elif falling_prob > 0.7:  # ML is VERY confident about fall
            predicted_class = "falling"
            confidence = falling_prob
            decision_reason = "ML very confident about fall"
        elif fall_score > 0.7 and falling_prob > 0.1:  # Strong physical + some ML evidence
            predicted_class = "falling"
            confidence = min(0.85, falling_prob + fall_score * 0.3)
            decision_reason = "Strong fall pattern + ML evidence"
        else:
            # Trust the ML model's top choice
            predicted_class = ml_top_choice
            confidence = ml_confidence
            decision_reason = f"ML classification: {ml_top_choice} (confidence: {ml_confidence:.3f})"
        
        # Get prediction probabilities for all classes
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        # Calculate fall indicators
        fall_indicators = {
            "high_impact": max_acc > 15.0,
            "free_fall": min_acc < 3.0,
            "sudden_change": max_change > 8.0,
            "high_rotation": max_gyro > 3.0,
            "strong_fall_pattern": fall_score > 0.5,
            "extreme_acceleration": max_acc > 20.0,
            "impact_detected": impact_score > 8.0,
            "high_statistical_variance": acc_std_50_max > 2.0 or gyro_std_50_max > 1.0
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": min(confidence, 1.0),
            "decision_reason": decision_reason,
            "ml_predictions": prediction_probs,
            "change_analysis": {
                "max_acceleration": round(max_acc, 3),
                "min_acceleration": round(min_acc, 3),
                "max_change_rate": round(max_change, 3),
                "acceleration_std": round(acc_std, 3),
                "max_gyro_velocity": round(max_gyro, 3),
                "impact_score": round(impact_score, 3),
                "fall_pattern_score": round(fall_score, 3),
                "axis_changes": {
                    "ax_max_change": round(ax_change, 3),
                    "ay_max_change": round(ay_change, 3),
                    "az_max_change": round(az_change, 3),
                    "wx_max_change": round(wx_change, 3),
                    "wy_max_change": round(wy_change, 3),
                    "wz_max_change": round(wz_change, 3)
                },
                "statistical_features": {
                    "acc_std_50_max": round(acc_std_50_max, 3),
                    "gyro_std_50_max": round(gyro_std_50_max, 3)
                }
            },
            "fall_indicators": fall_indicators,
            "model_info": {
                "model_type": "Anti-overfitting Enhanced GRU",
                "features_used": expected_features,
                "feature_breakdown": "12 base + 15 change detection = 27 total"
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
    Alternative endpoint for already preprocessed enhanced data.
    Expects 100 time steps with 28 features each.
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
        
        # Extract probabilities
        falling_prob = float(prediction[0][0])
        kneeling_prob = float(prediction[0][1])
        walking_prob = float(prediction[0][2])
        
        # Apply simplified decision logic for raw data
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # Enhanced logic for raw data (less detailed than full preprocessing)
        if falling_prob > 0.15:
            predicted_class = "falling"
            confidence = falling_prob
            decision_reason = "ML fall detection (raw data mode)"
        else:
            decision_reason = f"ML classification: {predicted_class} (raw data mode)"
        
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": min(confidence, 1.0),
            "decision_reason": decision_reason,
            "ml_predictions": prediction_probs,
            "note": "Raw data mode - detailed change analysis not available",
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
        "message": "Welcome to the Enhanced Anti-Overfitting Fall Detection API!",
        "model_loaded": model is not None,
        "model_info": {
            "type": "Anti-overfitting Enhanced GRU with Change Detection",
            "training_approach": "Realistic accuracy with anti-overfitting measures", 
            "features_per_timestep": 27,
            "feature_breakdown": "12 base + 15 change detection = 27 total"
        },
        "features": [
            "✓ Anti-overfitting training with realistic accuracy",
            "✓ ML-based classification with Bidirectional GRU",
            "✓ Enhanced change detection (acceleration & gyroscope)",
            "✓ Rolling window statistical features",
            "✓ Free fall pattern recognition",
            "✓ Impact detection with improved sensitivity",
            "✓ Multi-axis rotation analysis",
            "✓ Combined rule-based enhancement",
            "✓ Detailed change analysis with 16 features",
            "✓ Decision reasoning explanation"
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
        "model_type": "Enhanced Anti-Overfitting Fall Detection",
        "expected_performance": "Improved real-world accuracy with anti-overfitting training"
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
                    "enhanced_fall_pattern_score"
                ],
                "total_features_per_timestep": 27
            },
            "model_improvements": {
                "anti_overfitting": "✓ Aggressive regularization and early stopping",
                "realistic_accuracy": "✓ Trained for generalization, not memorization",
                "enhanced_features": "✓ 28 features including rolling statistics",
                "change_detection": "✓ 16 change detection features",
                "combined_approach": "✓ ML + enhanced rule-based decisions"
            },
            "fall_detection_thresholds": {
                "high_impact": "> 15 m/s²",
                "free_fall": "< 3 m/s²",
                "sudden_change": "> 8 m/s²",
                "high_rotation": "> 3 rad/s",
                "strong_fall_pattern": "> 0.5 score",
                "extreme_conditions": "> 0.9 fall score + > 30 m/s²"
            }
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Enhanced Anti-Overfitting Fall Detection API...")
    print("Model features:")
    print("- ✓ Anti-overfitting training with realistic accuracy")
    print("- ✓ Enhanced change detection (16 features)")
    print("- ✓ Rolling window statistics")
    print("- ✓ Physics-based fall pattern recognition")
    print("- ✓ Combined ML + rule-based decisions")
    print("- ✓ Comprehensive analysis and reasoning")
    print("- ✓ Improved error handling and validation")
    uvicorn.run(app, host="0.0.0.0", port=8000)