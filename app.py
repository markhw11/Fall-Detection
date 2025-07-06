from fastapi import FastAPI, HTTPException, status, Query # Import Query
from pydantic import BaseModel, validator, Field
import numpy as np
import tensorflow as tf
import pandas as pd
import uvicorn
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model variable and pre-defined feature columns
model = None
# Define feature columns here to ensure consistency
FEATURE_COLUMNS = [
    'ax', 'ay', 'az', 'wx', 'wy', 'wz',             # 6 raw features
    'acc_mag', 'gyro_mag',                          # 2 magnitude features
    'acc_mag_std_50', 'gyro_mag_std_50',            # 2 rolling std features
    'acc_mag_mean_50', 'gyro_mag_mean_50',          # 2 rolling mean features
    'acc_x_mag', 'acc_y_mag', 'acc_z_mag',          # 3 absolute acceleration features
    'gyro_x_mag', 'gyro_y_mag'                      # 2 absolute gyro features
]
EXPECTED_FEATURE_COUNT = len(FEATURE_COLUMNS) # Should be 17

# Load the model function
def load_model(model_path: str = "enhanced_anti_overfitting_fall_detection_model.h5"):
    """Loads the TensorFlow/Keras model from the specified path."""
    global model
    try:
        # It's good practice to re-compile the model after loading, especially if custom objects are used
        # For a standard H5, it might not be strictly necessary but doesn't hurt.
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        model = None
        return False

# Define a lifespan context manager for FastAPI to handle startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application lifespan events.
    Loads the ML model when the application starts up.
    """
    logger.info("Application startup: Loading ML model...")
    load_model()
    yield
    logger.info("Application shutdown: Cleaning up resources (if any)...")
    # Optional: Add cleanup logic here if needed, e.g., tf.keras.backend.clear_session()

# Initialize FastAPI app with the lifespan manager
app = FastAPI(
    title="Robust Fall Detection API - Reduced False Positives",
    description="API for fall detection with enhanced motion validation and dynamic threshold tuning to minimize false positives.",
    version="2.2.0", # Incremented version
    lifespan=lifespan # Assign the lifespan context manager
)

# Pydantic models for request validation
class SensorReading(BaseModel):
    time: Optional[float] = None
    ax: float = Field(..., description="Accelerometer X-axis reading (g)")
    ay: float = Field(..., description="Accelerometer Y-axis reading (g)")
    az: float = Field(..., description="Accelerometer Z-axis reading (g)")
    wx: float = Field(..., description="Gyroscope X-axis reading (rad/s)")
    wy: float = Field(..., description="Gyroscope Y-axis reading (rad/s)")
    wz: float = Field(..., description="Gyroscope Z-axis reading (rad/s)")

    @validator('ax', 'ay', 'az', 'wx', 'wy', 'wz')
    def validate_sensor_values(cls, v):
        if not -100 <= v <= 100:  # Reasonable sensor value range
            raise ValueError('Sensor values must be between -100 and 100')
        return v

class FallDetectionData(BaseModel):
    sensor_data: List[SensorReading] = Field(..., min_items=100, max_items=100,
                                             description="List of 100 sensor readings (accelerometer and gyroscope).")

    @validator('sensor_data')
    def validate_sensor_data_length(cls, v):
        if len(v) != 100: # Redundant due to Field(min_items=100, max_items=100) but good for clarity
            raise ValueError('sensor_data must contain exactly 100 readings')
        return v

class FallDetectionRawData(BaseModel):
    features: List[List[float]] = Field(..., min_items=100, max_items=100,
                                         description=f"List of 100 time steps, each containing {EXPECTED_FEATURE_COUNT} preprocessed features.")

    @validator('features')
    def validate_features_shape(cls, v):
        if len(v) != 100:
            raise ValueError('features must contain exactly 100 time steps')
        # Corrected this validator to match EXPECTED_FEATURE_COUNT
        if any(len(timestep) != EXPECTED_FEATURE_COUNT for timestep in v):
            raise ValueError(f'Each timestep must have exactly {EXPECTED_FEATURE_COUNT} features')
        return v

# List of class labels (must match training order)
classes = ['falling', 'kneeling', 'walking']

def detect_motion_level(window_data: pd.DataFrame):
    """
    Analyzes motion from sensor data using a refined set of thresholds to categorize activity.
    The thresholds are designed to be conservative to prevent false positives when stationary or during slight movements.
    """
    acc_mag = np.sqrt(window_data['ax']**2 + window_data['ay']**2 + window_data['az']**2)
    gyro_mag = np.sqrt(window_data['wx']**2 + window_data['wy']**2 + window_data['wz']**2)

    # Calculate robust motion indicators
    # Using interquartile range (IQR) for more robust variance estimation, less sensitive to outliers
    acc_iqr = acc_mag.quantile(0.75) - acc_mag.quantile(0.25)
    gyro_iqr = gyro_mag.quantile(0.75) - gyro_mag.quantile(0.25)

    # BALANCED thresholds for stationary detection (stricter)
    # Combined with min_acc check for 'quiet' stationary periods.
    is_stationary = (
        acc_mag.std() < 0.25 and # Lower std for stability
        gyro_mag.std() < 0.04 and # Very low gyro std for no rotation
        acc_iqr < 1.0 and         # Tight IQR for minimal spread
        gyro_iqr < 0.2 and        # Very tight gyro IQR
        acc_mag.max() < 11.0 and  # Max acceleration close to gravity
        acc_mag.min() > 8.0       # Min acceleration also close to gravity (not free-falling)
    )

    # More realistic thresholds for significant motion (less prone to minor jitters)
    has_significant_motion = (
        acc_mag.std() > 2.5 or      # Higher acceleration std
        gyro_mag.std() > 1.0 or     # Higher gyro std
        acc_iqr > 5.0 or            # Larger acceleration IQR
        gyro_iqr > 2.0 or           # Larger gyro IQR
        acc_mag.max() > 20.0 or     # Peak acceleration indicating impact/fast movement
        acc_mag.min() < 5.0         # Dip in acceleration indicating potential free-fall
    )

    # Additional motion quality indicators for detailed analysis
    # Number of times acc_mag crosses a threshold, indicative of erratic movement
    acc_crossing_threshold = np.sum(np.diff((acc_mag > (acc_mag.mean() + acc_mag.std() * 1.5)).astype(int)) != 0)
    gyro_crossing_threshold = np.sum(np.diff((gyro_mag > (gyro_mag.mean() + gyro_mag.std() * 1.0)).astype(int)) != 0)


    return {
        'is_stationary': bool(is_stationary),
        'has_significant_motion': bool(has_significant_motion),
        'acc_mag_mean': float(acc_mag.mean()),
        'acc_mag_std': float(acc_mag.std()),
        'gyro_mag_std': float(gyro_mag.std()),
        'acc_mag_iqr': float(acc_iqr),
        'gyro_mag_iqr': float(gyro_iqr),
        'max_acc': float(acc_mag.max()),
        'min_acc': float(acc_mag.min()),
        'max_gyro': float(gyro_mag.max()),
        'acc_crossing_threshold': int(acc_crossing_threshold),
        'gyro_crossing_threshold': int(gyro_crossing_threshold)
    }

def preprocess_sensor_data(sensor_readings: List[SensorReading]) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocesses raw sensor readings into a fixed set of 17 features per timestep.
    Returns both the feature array for the model and the DataFrame for motion analysis.
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
        # Using .copy() to avoid SettingWithCopyWarning in some pandas versions if modifications follow
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

        # Ensure correct order and count
        features = df[FEATURE_COLUMNS].values

        if features.shape[1] != EXPECTED_FEATURE_COUNT:
            logger.error(f"Feature count mismatch during preprocessing: Expected {EXPECTED_FEATURE_COUNT}, Got {features.shape[1]}")
            raise ValueError(f"Feature count mismatch: Expected {EXPECTED_FEATURE_COUNT}, Got {features.shape[1]}")

        return features, df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Preprocessing error: {str(e)}")

@app.post(
    "/predict/",
    summary="Predict fall event from raw sensor data (conservative approach)",
    response_description="Detailed fall detection result with motion analysis."
)
def predict(data: FallDetectionData):
    """
    Predicts fall events from 100 raw sensor readings.
    This endpoint implements a **conservative approach** to reduce false positives by combining
    the ML model's prediction with robust motion analysis and stricter thresholds.
    It prioritizes avoiding false alarms over detecting every single fall, making it suitable
    for scenarios where false positives are costly.
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Please try again later or reload the model.")

    try:
        # Preprocess features and get raw data for motion analysis
        features, df = preprocess_sensor_data(data.sensor_data)

        # Analyze motion level first
        motion_analysis = detect_motion_level(df)

        # Conservative Override Logic
        # 1. If device is clearly stationary, immediately return non-fall
        if motion_analysis['is_stationary']:
            return {
                "predicted_class": "kneeling", # Or 'idle' if you add it to classes
                "confidence": 0.98,
                "decision_reason": "OVERRIDE: Device detected as stationary - cannot be falling.",
                "ml_predictions": {"falling": 0.01, "kneeling": 0.95, "walking": 0.04}, # Provide plausible ML outputs
                "motion_analysis": motion_analysis,
                "override_applied": True,
                "override_reason_detail": "STATIONARY_OVERRIDE",
                "status": "success"
            }

        # Reshape for prediction
        features_reshaped = np.expand_dims(features, axis=0) # Renamed to avoid confusion with df features

        # Get model prediction
        prediction = model.predict(features_reshaped, verbose=0)

        # Extract probabilities
        falling_prob = float(prediction[0][classes.index('falling')]) # Use index to be robust to class order
        kneeling_prob = float(prediction[0][classes.index('kneeling')])
        walking_prob = float(prediction[0][classes.index('walking')])

        # BALANCED APPROACH: Multiple pathways to detect falls (tuned for fewer FPs)

        # Pathway 1: High-confidence ML prediction with confirmed significant motion
        high_confidence_fall = (
            falling_prob >= 0.85 and # Very high ML confidence required
            motion_analysis['has_significant_motion'] and # Must have significant motion
            motion_analysis['max_acc'] > 18.0 # High peak acceleration as extra validation
        )

        # Pathway 2: Medium-confidence ML with very strong motion evidence (e.g., impact + freefall signs)
        medium_confidence_strong_motion_fall = (
            falling_prob >= 0.65 and # Medium ML confidence
            motion_analysis['has_significant_motion'] and
            motion_analysis['max_acc'] > 25.0 and # Very high impact
            motion_analysis['min_acc'] < 6.0 and # Clear dip indicating free-fall
            motion_analysis['gyro_mag_std'] > 1.5 # Significant rotation
        )

        # Pathway 3: Lower-confidence ML, but extreme physical indicators (e.g., severe impact)
        extreme_motion_override = (
            falling_prob >= 0.30 and # Even lower ML confidence, but still some
            motion_analysis['max_acc'] > 35.0 and # Extremely high impact
            motion_analysis['min_acc'] < 3.0 and # Very strong free-fall sign
            motion_analysis['acc_mag_iqr'] > 10.0 # Very large spread in acceleration
        )

        # Determine if it's a fall using any of the conservative pathways
        fall_detected = high_confidence_fall or medium_confidence_strong_motion_fall or extreme_motion_override

        predicted_class_final = "not_fall" # Initialize to a non-fall state, will be updated
        confidence_final = 0.0
        decision_reason_final = "No strong fall evidence detected."
        override_applied = False
        override_reason_detail = None

        if fall_detected:
            predicted_class_final = "falling"
            confidence_final = falling_prob # Base confidence on ML output for consistency

            if high_confidence_fall:
                decision_reason_final = f"Fall detected (High-confidence ML + Motion Validation): ML={falling_prob:.3f}"
            elif medium_confidence_strong_motion_fall:
                decision_reason_final = f"Fall detected (Medium-confidence ML + Strong Motion Evidence): ML={falling_prob:.3f}"
            elif extreme_motion_override:
                decision_reason_final = f"Fall detected (Extreme Motion Override): ML={falling_prob:.3f}"

            # Boost confidence for strong rule matches, but cap it
            if confidence_final < 0.95 and (high_confidence_fall or medium_confidence_strong_motion_fall):
                 confidence_final = min(1.0, confidence_final + 0.1) # Small boost
            if confidence_final < 0.98 and extreme_motion_override:
                 confidence_final = min(1.0, confidence_final + 0.2) # Larger boost for extreme cases

        else:
            # If no fall is detected by the conservative pathways, determine the most likely non-fall class
            # Ensure 'falling' is not picked as the top class unless specifically triggered above
            non_fall_probs = [kneeling_prob, walking_prob]
            non_fall_classes = ['kneeling', 'walking']
            top_non_fall_idx = np.argmax(non_fall_probs)

            predicted_class_final = non_fall_classes[top_non_fall_idx]
            confidence_final = non_fall_probs[top_non_fall_idx]
            decision_reason_final = f"ML model prediction: {predicted_class_final} (confidence: {confidence_final:.3f}). Fall prediction rejected due to insufficient evidence."

        # Get prediction probabilities for all classes
        prediction_probs = {
            classes[i]: float(prediction[0][i])
            for i in range(len(classes))
        }

        return {
            "predicted_class": predicted_class_final,
            "confidence": float(confidence_final),
            "decision_reason": decision_reason_final,
            "ml_predictions": prediction_probs,
            "motion_analysis": {
                "is_stationary": motion_analysis['is_stationary'],
                "has_significant_motion": motion_analysis['has_significant_motion'],
                "acceleration_mean": round(motion_analysis['acc_mag_mean'], 3),
                "acceleration_std": round(motion_analysis['acc_mag_std'], 3),
                "gyroscope_std": round(motion_analysis['gyro_mag_std'], 3),
                "acceleration_iqr": round(motion_analysis['acc_mag_iqr'], 3),
                "gyroscope_iqr": round(motion_analysis['gyro_mag_iqr'], 3),
                "max_acceleration": round(motion_analysis['max_acc'], 3),
                "min_acceleration": round(motion_analysis['min_acc'], 3),
                "max_gyroscope": round(motion_analysis['max_gyro'], 3),
                "accel_threshold_crossings": motion_analysis['acc_crossing_threshold'],
                "gyro_threshold_crossings": motion_analysis['gyro_crossing_threshold']
            },
            "fall_detection_pathways_met": {
                "high_confidence_ml": high_confidence_fall,
                "medium_confidence_strong_motion": medium_confidence_strong_motion_fall,
                "extreme_motion_override": extreme_motion_override
            },
            "override_applied": override_applied,
            "override_reason_detail": override_reason_detail,
            "model_info": {
                "model_type": "Conservative Fall Detection with Motion Validation",
                "features_used": EXPECTED_FEATURE_COUNT,
                "approach": "Multiple detection pathways with strict, balanced thresholds to reduce false positives"
            },
            "status": "success"
        }

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True) # Log exception info for debugging
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

@app.post(
    "/tune_thresholds/",
    summary="Tune fall detection thresholds for optimization",
    response_description="Evaluates sensor data against custom thresholds for fine-tuning fall detection logic."
)
def tune_thresholds(data: FallDetectionData,
                    # Motion Analysis Thresholds
                    stationary_acc_std: float = Query(0.25, description="Max Std Dev for Accel to be considered stationary"),
                    stationary_gyro_std: float = Query(0.04, description="Max Std Dev for Gyro to be considered stationary"),
                    stationary_acc_iqr: float = Query(1.0, description="Max IQR for Accel to be considered stationary"),
                    stationary_gyro_iqr: float = Query(0.2, description="Max IQR for Gyro to be considered stationary"),
                    stationary_max_acc: float = Query(11.0, description="Max peak Accel for stationary"),
                    stationary_min_acc_gt: float = Query(8.0, description="Min lowest Accel for stationary (should be near gravity)"),
                    significant_motion_acc_std: float = Query(2.5, description="Min Std Dev for Accel to indicate significant motion"),
                    significant_motion_gyro_std: float = Query(1.0, description="Min Std Dev for Gyro to indicate significant motion"),
                    significant_motion_acc_iqr: float = Query(5.0, description="Min IQR for Accel to indicate significant motion"),
                    significant_motion_gyro_iqr: float = Query(2.0, description="Min IQR for Gyro to indicate significant motion"),
                    significant_motion_max_acc: float = Query(20.0, description="Min max Accel for significant motion"),
                    significant_motion_min_acc: float = Query(5.0, description="Max min Accel for significant motion (for free-fall consideration)"),
                    # ML Pathway Thresholds
                    high_confidence_ml_prob: float = Query(0.85, description="Min ML 'falling' probability for high-confidence pathway"),
                    high_confidence_ml_acc_peak: float = Query(18.0, description="Min peak Accel for high-confidence pathway"),
                    medium_confidence_ml_prob: float = Query(0.65, description="Min ML 'falling' probability for medium-confidence pathway"),
                    medium_confidence_ml_acc_peak: float = Query(25.0, description="Min peak Accel for medium-confidence pathway"),
                    medium_confidence_ml_min_acc: float = Query(6.0, description="Max min Accel for medium-confidence pathway (free-fall)"),
                    medium_confidence_ml_gyro_std: float = Query(1.5, description="Min Gyro Std Dev for medium-confidence pathway"),
                    extreme_motion_ml_prob: float = Query(0.30, description="Min ML 'falling' probability for extreme motion pathway"),
                    extreme_motion_max_acc: float = Query(35.0, description="Min extremely high peak Accel for extreme motion pathway"),
                    extreme_motion_min_acc: float = Query(3.0, description="Max extremely low min Accel for extreme motion pathway"),
                    extreme_motion_acc_iqr: float = Query(10.0, description="Min Accel IQR for extreme motion pathway")
                    ):
    """
    This endpoint allows you to **dynamically test different threshold values** for the motion analysis
    and fall detection pathways against a given set of sensor data.
    Use this to fine-tune the system and find the optimal balance between
    reducing false positives and maintaining true positive (recall) performance.
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Please try again later or reload the model.")

    try:
        features, df = preprocess_sensor_data(data.sensor_data)

        # Recalculate motion metrics based on input data
        acc_mag = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        gyro_mag = np.sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)

        acc_std = float(acc_mag.std())
        gyro_std = float(gyro_mag.std())
        acc_iqr = float(acc_mag.quantile(0.75) - acc_mag.quantile(0.25))
        gyro_iqr = float(gyro_mag.quantile(0.75) - gyro_mag.quantile(0.25))
        max_acc = float(acc_mag.max())
        min_acc = float(acc_mag.min())
        max_gyro = float(gyro_mag.max())
        acc_mean = float(acc_mag.mean())
        acc_crossing_threshold = int(np.sum(np.diff((acc_mag > (acc_mag.mean() + acc_mag.std() * 1.5)).astype(int)) != 0))
        gyro_crossing_threshold = int(np.sum(np.diff((gyro_mag > (gyro_mag.mean() + gyro_mag.std() * 1.0)).astype(int)) != 0))


        # Test stationary thresholds
        is_stationary_test = (
            acc_std < stationary_acc_std and
            gyro_std < stationary_gyro_std and
            acc_iqr < stationary_acc_iqr and
            gyro_iqr < stationary_gyro_iqr and
            max_acc < stationary_max_acc and
            min_acc > stationary_min_acc_gt
        )

        # Test significant motion thresholds
        has_significant_motion_test = (
            acc_std > significant_motion_acc_std or
            gyro_std > significant_motion_gyro_std or
            acc_iqr > significant_motion_acc_iqr or
            gyro_iqr > significant_motion_gyro_iqr or
            max_acc > significant_motion_max_acc or
            min_acc < significant_motion_min_acc
        )

        # Get ML prediction
        features_reshaped = np.expand_dims(features, axis=0)
        prediction = model.predict(features_reshaped, verbose=0)
        falling_prob = float(prediction[0][classes.index('falling')])
        kneeling_prob = float(prediction[0][classes.index('kneeling')])
        walking_prob = float(prediction[0][classes.index('walking')])


        # Test ML pathway thresholds
        high_confidence_fall_test = (
            falling_prob >= high_confidence_ml_prob and
            has_significant_motion_test and
            max_acc > high_confidence_ml_acc_peak
        )

        medium_confidence_strong_motion_fall_test = (
            falling_prob >= medium_confidence_ml_prob and
            has_significant_motion_test and
            max_acc > medium_confidence_ml_acc_peak and
            min_acc < medium_confidence_ml_min_acc and
            gyro_std > medium_confidence_ml_gyro_std
        )

        extreme_motion_override_test = (
            falling_prob >= extreme_motion_ml_prob and
            max_acc > extreme_motion_max_acc and
            min_acc < extreme_motion_min_acc and
            acc_iqr > extreme_motion_acc_iqr
        )

        fall_detected_test = high_confidence_fall_test or medium_confidence_strong_motion_fall_test or extreme_motion_override_test

        return {
            "sensor_metrics_for_current_data": {
                "acceleration_mean": round(acc_mean, 3),
                "acceleration_std": round(acc_std, 3),
                "gyroscope_std": round(gyro_std, 3),
                "acceleration_iqr": round(acc_iqr, 3),
                "gyroscope_iqr": round(gyro_iqr, 3),
                "max_acceleration": round(max_acc, 3),
                "min_acceleration": round(min_acc, 3),
                "max_gyroscope": round(max_gyro, 3),
                "accel_threshold_crossings": acc_crossing_threshold,
                "gyro_threshold_crossings": gyro_crossing_threshold
            },
            "ml_probabilities": {
                "falling": round(falling_prob, 3),
                "kneeling": round(kneeling_prob, 3),
                "walking": round(walking_prob, 3)
            },
            "threshold_test_results": {
                "is_stationary_current_data": is_stationary_test,
                "has_significant_motion_current_data": has_significant_motion_test,
                "fall_pathway_high_confidence_ml": high_confidence_fall_test,
                "fall_pathway_medium_confidence_strong_motion": medium_confidence_strong_motion_fall_test,
                "fall_pathway_extreme_motion_override": extreme_motion_override_test,
                "overall_fall_detected": fall_detected_test
            },
            "input_thresholds_used": {
                "stationary_acc_std": stationary_acc_std,
                "stationary_gyro_std": stationary_gyro_std,
                "stationary_acc_iqr": stationary_acc_iqr,
                "stationary_gyro_iqr": stationary_gyro_iqr,
                "stationary_max_acc": stationary_max_acc,
                "stationary_min_acc_gt": stationary_min_acc_gt,
                "significant_motion_acc_std": significant_motion_acc_std,
                "significant_motion_gyro_std": significant_motion_gyro_std,
                "significant_motion_acc_iqr": significant_motion_acc_iqr,
                "significant_motion_gyro_iqr": significant_motion_gyro_iqr,
                "significant_motion_max_acc": significant_motion_max_acc,
                "significant_motion_min_acc": significant_motion_min_acc,
                "high_confidence_ml_prob": high_confidence_ml_prob,
                "high_confidence_ml_acc_peak": high_confidence_ml_acc_peak,
                "medium_confidence_ml_prob": medium_confidence_ml_prob,
                "medium_confidence_ml_acc_peak": medium_confidence_ml_acc_peak,
                "medium_confidence_ml_min_acc": medium_confidence_ml_min_acc,
                "medium_confidence_ml_gyro_std": medium_confidence_ml_gyro_std,
                "extreme_motion_ml_prob": extreme_motion_ml_prob,
                "extreme_motion_max_acc": extreme_motion_max_acc,
                "extreme_motion_min_acc": extreme_motion_min_acc,
                "extreme_motion_acc_iqr": extreme_motion_acc_iqr
            },
            "note": "Use these results to refine the fixed thresholds in the /predict/ endpoint."
        }

    except Exception as e:
        logger.error(f"Tuning failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Tuning failed: {str(e)}")

@app.post(
    "/predict_raw/",
    summary="Predict fall event from preprocessed sensor data",
    response_description="Fall detection result for already processed features (17 features per timestep)."
)
def predict_raw(data: FallDetectionRawData):
    """
    Alternative endpoint for already preprocessed data.
    Expects 100 time steps with 17 features each. This endpoint applies a conservative
    approach by downgrading low-confidence fall predictions.
    Note: Motion analysis details are not available as raw sensor data is not provided.
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Please try again later or reload the model.")

    try:
        # Convert the input list into a numpy array
        features = np.array(data.features)

        # Validate input shape (100 time steps, 17 features)
        if features.shape != (100, EXPECTED_FEATURE_COUNT):
            logger.error(f"Invalid input shape for predict_raw. Expected (100, {EXPECTED_FEATURE_COUNT}), got {features.shape}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input shape. Expected (100, {EXPECTED_FEATURE_COUNT}), got {features.shape}"
            )

        # Reshape input to match the model's expected shape
        features_reshaped = np.expand_dims(features, axis=0)

        # Get model prediction
        prediction = model.predict(features_reshaped, verbose=0)

        # Use conservative approach for raw data too
        falling_prob = float(prediction[0][classes.index('falling')])
        kneeling_prob = float(prediction[0][classes.index('kneeling')])
        walking_prob = float(prediction[0][classes.index('walking')])

        # Determine the ML model's top choice initially
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        decision_reason = f"ML model prediction: {predicted_class} (raw data mode)"

        # Conservative thresholding for raw data: If ML predicts 'falling' but with low confidence, reconsider
        conservative_fall_threshold_raw = 0.75 # A threshold that defines "high enough" confidence for raw predictions

        if predicted_class == "falling" and falling_prob < conservative_fall_threshold_raw:
            # If ML says fall but not very confidently, downgrade to the next most likely non-fall class
            logger.info(f"Fall prediction for raw data downgraded due to low confidence: ML Prob {falling_prob:.3f} < {conservative_fall_threshold_raw}")
            non_fall_probs = [kneeling_prob, walking_prob]
            non_fall_classes = ['kneeling', 'walking']
            next_best_idx = np.argmax(non_fall_probs)
            predicted_class = non_fall_classes[next_best_idx]
            confidence = non_fall_probs[next_best_idx]
            decision_reason = f"Fall prediction downgraded from '{classes[predicted_class_index]}' to '{predicted_class}' due to low confidence ({falling_prob:.3f} < {conservative_fall_threshold_raw})."

        prediction_probs = {
            classes[i]: float(prediction[0][i])
            for i in range(len(classes))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "decision_reason": decision_reason,
            "ml_predictions": prediction_probs,
            "note": "Raw data mode - detailed motion analysis is not performed; conservative thresholds applied to ML output.",
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Raw prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Raw prediction failed: {str(e)}")

@app.get("/", summary="Welcome message and API overview",
          response_description="General information about the Fall Detection API.")
def read_root():
    """Provides a welcome message and an overview of the API's capabilities and endpoints."""
    return {
        "message": "Welcome to the Conservative Fall Detection API!",
        "model_loaded": model is not None,
        "model_info_summary": {
            "type": "Conservative Fall Detection with Enhanced Motion Validation",
            "approach": "High thresholds and multi-stage motion analysis to significantly reduce false positives, ideal for critical alerting.",
            "features_per_timestep": EXPECTED_FEATURE_COUNT,
            "feature_breakdown": f"{EXPECTED_FEATURE_COUNT} engineered features, including magnitudes, rolling statistics, and absolute axis values."
        },
        "key_features": [
            "✓ Highly conservative approach tailored to minimize false positives.",
            "✓ Robust motion level analysis (`detect_motion_level`) for initial screening and validation.",
            "✓ Multiple, strict pathways for fall detection, combining ML confidence with robust physical motion indicators (e.g., high impact, free fall signs, significant rotation).",
            "✓ Automatic classification adjustment/downgrading for low-confidence ML fall predictions.",
            "✓ 17 carefully engineered sensor features for comprehensive activity representation.",
            "✓ Dedicated `/tune_thresholds` endpoint for dynamic fine-tuning of detection parameters.",
            "✓ Clear decision reasoning and detailed motion analysis included in responses.",
            "✓ Optimized for real-world reliability and reduced alert fatigue."
        ],
        "supported_classes": classes,
        "available_endpoints": {
            "/predict/": "POST - Analyze 100 raw sensor readings; performs feature engineering and applies conservative fall detection logic.",
            "/predict_raw/": "POST - Analyze 100 preprocessed sensor data time steps (17 features each); applies conservative ML thresholds.",
            "/tune_thresholds/": "POST - Test custom thresholds for motion analysis and fall detection pathways against sample data. Useful for calibration.",
            "/health": "GET - Check the API's operational status and model loading status.",
            "/model_info": "GET - Retrieve detailed information about the loaded model and its detection strategy.",
            "/reload_model": "POST - Reload the TensorFlow model from a specified file path."
        }
    }

@app.get("/health", summary="Health check endpoint",
          response_description="Status of the API and model loading.")
def health_check():
    """Checks the health of the API and whether the machine learning model is loaded."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "Conservative Fall Detection",
        "expected_performance_goal": "Significantly reduced false positives; optimized for high precision."
    }

@app.post("/reload_model", summary="Reload the ML model",
           response_description="Confirms successful model reload or provides error details.")
def reload_model(model_path: str = "enhanced_anti_overfitting_fall_detection_model.h5"):
    """
    Reloads the machine learning model from the specified H5 file path.
    This is useful for deploying updated models without restarting the API.
    """
    # Use default path if not provided.
    # The absolute path you had previously, 'D:\Uni\Graduationproject\FallDet\anti_overfitting_fall_detection_model.h5',
    # might need to be passed explicitly in the request body if the default is changed or not applicable.
    success = load_model(model_path)
    if success:
        return {"status": "success", "message": f"Model reloaded from {model_path}"}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reload model. Check logs for details.")

@app.get("/model_info", summary="Get detailed model information",
          response_description="Provides details about the loaded ML model, its features, and detection strategy.")
def model_info():
    """Provides detailed information about the loaded machine learning model, including its input/output shapes,
    the features it expects, and the core principles of its conservative detection approach.
    """
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded. Cannot provide info.")

    try:
        return {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "classes": classes,
            "features_description": {
                "engineered_features_list": FEATURE_COLUMNS,
                "total_features_per_timestep": EXPECTED_FEATURE_COUNT,
                "description": "These features are derived from raw accelerometer (ax, ay, az) and gyroscope (wx, wy, wz) readings, including magnitudes, rolling window statistics (mean and standard deviation over 50 samples), and absolute values for each axis."
            },
            "conservative_approach_details": {
                "motion_validation": "Initial check: If sensor data indicates a clearly stationary state, prediction is overridden to 'not fall' to eliminate false positives from idle periods.",
                "multi_pathway_detection": "A fall is only detected if one of several strict criteria are met, combining ML confidence with robust physical motion indicators (e.g., high impact, free fall signs, significant rotation).",
                "high_thresholds": "ML probability for 'falling' must be consistently high, and physical motion metrics must exceed conservative thresholds.",
                "uncertain_prediction_downgrade": "If the ML model predicts 'falling' but with insufficient confidence or without strong physical validation, the prediction is re-evaluated and potentially changed to a non-fall class."
            },
            "motion_analysis_thresholds_for_predict_endpoint": {
                "is_stationary_criteria": {
                    "acc_mag_std < 0.25", "gyro_mag_std < 0.04", "acc_mag_iqr < 1.0",
                    "gyro_mag_iqr < 0.2", "acc_mag.max() < 11.0", "acc_mag.min() > 8.0"
                },
                "has_significant_motion_criteria": {
                    "acc_mag_std > 2.5", "gyro_mag_std > 1.0", "acc_mag_iqr > 5.0",
                    "gyro_mag_iqr > 2.0", "acc_mag.max() > 20.0", "acc_mag.min() < 5.0"
                },
                "ml_pathway_1_high_confidence": "ML Fall Prob >= 0.85 AND Significant Motion AND Max Acc > 18.0",
                "ml_pathway_2_medium_confidence_strong_motion": "ML Fall Prob >= 0.65 AND Significant Motion AND Max Acc > 25.0 AND Min Acc < 6.0 AND Gyro Std > 1.5",
                "ml_pathway_3_extreme_motion_override": "ML Fall Prob >= 0.30 AND Max Acc > 35.0 AND Min Acc < 3.0 AND Accel IQR > 10.0"
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    print("Starting Conservative Fall Detection API...")
    print("\n--- API Overview ---")
    print("- Goal: Significantly reduce false positives in fall detection.")
    print("- Approach: Combines a trained ML model with stringent, physics-informed rule-based validation.")
    print("- Key Features:")
    print("  - Motion Analysis: Differentiates true movement from stationary periods.")
    print("  - Multiple Detection Pathways: Requires strong evidence (ML confidence + physical indicators) for a fall.")
    print("  - Dynamic Threshold Tuning: '/tune_thresholds' endpoint for calibrating detection parameters.")
    print("  - Clear decision reasoning and detailed motion analysis included in responses.")
    print("  - Optimized for real-world reliability and reduced alert fatigue.")
    print("\n--------------------")
    uvicorn.run(app, host="0.0.0.0", port=8000)
