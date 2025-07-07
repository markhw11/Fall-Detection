from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
import uvicorn
from typing import List

# Initialize FastAPI app
app = FastAPI(title="Enhanced Anti-Overfitting Fall Detection API", description="API for fall detection with change detection and anti-overfitting")

# --- Configuration Constants (Centralized Thresholds for easier tuning) ---
# These values are derived from your notebook's training and testing analysis.
# Adjust these to fine-tune sensitivity and reduce false positives.

# Rule-based feature calculation thresholds (from calculate_change_features in notebook)
THRESHOLD_MAX_ACC_FALL_SCORE = 20.0  # Increased from 15.0
THRESHOLD_MIN_ACC_FALL_SCORE = 2.0   # Decreased from 3.0 (stricter free-fall)
THRESHOLD_ACC_DIFF_FALL_SCORE = 12.0 # Increased from 8.0
THRESHOLD_MAX_GYRO_FALL_SCORE = 5.0  # Increased from 3.0 (Now used more strictly)

THRESHOLD_IMPACT_PRE_ACC = 5.0      # Unchanged, as this is a pre-impact state
THRESHOLD_IMPACT_POST_ACC = 12.0    # Unchanged, as this is a post-impact state

# Hybrid decision logic thresholds (from predict endpoint in notebook's test function)
OVERRIDE_FALL_SCORE_HIGH = 0.95    # Increased from 0.9
OVERRIDE_MAX_ACC_EXTREME = 40.0    # Increased from 30.0
OVERRIDE_MIN_ACC_EXTREME = 0.5     # Decreased from 1.0 (stricter free-fall)

ML_CONF_FALL_HIGH = 0.7            # Unchanged (model rarely hits this anyway)
ML_CONF_FALL_MEDIUM = 0.6          # Adjusted from 0.5 to 0.6
ML_CONF_FALL_LOW = 0.55            # <--- LATEST ADJUSTMENT: Increased from 0.5

THRESHOLD_MAX_CHANGE_AMBIGUOUS = 15.0 # Increased from 10.0 (Higher threshold for generic sudden change)
THRESHOLD_MIN_ACC_AMBIGUOUS = 2.0    # Unchanged, but ML_CONF_FALL_MEDIUM now applies
THRESHOLD_IMPACT_AMBIGUOUS = 12.0    # Increased from 8.0, combined with ML_CONF_FALL_LOW

# NEW: Threshold for combining sudden change with rotation
THRESHOLD_MAX_CHANGE_ROTATION_COMBINED = 8.0 # Lower than above, but requires rotation

# Post-prediction confirmation (New concept for this problem)
CONFIRMATION_WINDOWS = 3 # Number of consecutive 'falling' predictions to confirm a fall.
STILLNESS_ACC_STD_THRESHOLD = 0.5 # Low std dev of acceleration for stillness
STILLNESS_GYRO_MAX_THRESHOLD = 0.5 # Low max gyro for stillness
STILLNESS_DURATION_WINDOWS = 2   # How many windows of stillness to consider

# --- Global Variables ---
fall_prediction_history = {} # Stores {'device_id': {'last_pred': 'str', 'count': int, 'last_timestamp': float}}

# Load the enhanced anti-overfitting trained model
try:
    model = tf.keras.models.load_model(r"Final.h5")
    print("Enhanced anti-overfitting model loaded successfully!")
    if model.input_shape[1:] != (100, 17):
        print(f"WARNING: Model input shape mismatch! Expected (100, 17), got {model.input_shape[1:]}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class to represent input data for prediction
class SensorReading(BaseModel):
    time: float
    ax: float
    ay: float
    az: float
    wx: float
    wy: float
    wz: float
    device_id: str = "default_device"

class FallDetectionData(BaseModel):
    sensor_data: List[SensorReading]

class FallDetectionRawData(BaseModel):
    features: List[List[float]]
    device_id: str = "default_device"

classes = ['falling', 'kneeling', 'walking']

def calculate_change_features(window_data: pd.DataFrame) -> np.ndarray:
    """
    Calculate drastic change indicators for a window of sensor data.
    Must match the training function exactly!
    """
    features_list = []
    
    acc_mag = np.sqrt(window_data['ax']**2 + window_data['ay']**2 + window_data['az']**2)
    
    max_acc = acc_mag.max()
    features_list.append(max_acc)
    
    acc_diff = acc_mag.diff().abs().max()
    features_list.append(acc_diff if not pd.isna(acc_diff) else 0)
    
    acc_std = acc_mag.std()
    features_list.append(acc_std if not pd.isna(acc_std) else 0)
    
    max_gyro = max(window_data['wx'].abs().max(), 
                   window_data['wy'].abs().max(), 
                   window_data['wz'].abs().max())
    features_list.append(max_gyro)
    
    min_acc = acc_mag.min()
    features_list.append(min_acc)
    
    impact_score = 0
    for i in range(1, len(acc_mag)):
        if acc_mag.iloc[i-1] < THRESHOLD_IMPACT_PRE_ACC and acc_mag.iloc[i] > THRESHOLD_IMPACT_POST_ACC:
            impact_score = max(impact_score, acc_mag.iloc[i] - acc_mag.iloc[i-1])
    features_list.append(impact_score)
    
    ax_change = window_data['ax'].diff().abs().max()
    ay_change = window_data['ay'].diff().abs().max()
    az_change = window_data['az'].diff().abs().max()
    features_list.extend([
        ax_change if not pd.isna(ax_change) else 0,
        ay_change if not pd.isna(ay_change) else 0,
        az_change if not pd.isna(az_change) else 0
    ])
    
    fall_score = 0
    if max_acc > THRESHOLD_MAX_ACC_FALL_SCORE:
        fall_score += 0.3
    if min_acc < THRESHOLD_MIN_ACC_FALL_SCORE:
        fall_score += 0.25
    if acc_diff > THRESHOLD_ACC_DIFF_FALL_SCORE:
        fall_score += 0.25
    if max_gyro > THRESHOLD_MAX_GYRO_FALL_SCORE:
        fall_score += 0.2
    features_list.append(fall_score)
    
    return np.array(features_list).reshape(1, -1)

def preprocess_enhanced_sensor_data(sensor_readings: List[SensorReading]) -> tuple:
    """
    Preprocess sensor data with change detection features for the enhanced model.
    """
    data_list = []
    for reading in sensor_readings:
        data_list.append([
            reading.ax, reading.ay, reading.az,
            reading.wx, reading.wy, reading.wz
        ])
    
    df = pd.DataFrame(data_list, columns=['ax', 'ay', 'az', 'wx', 'wy', 'wz'])
    
    if 'acc_mag' not in df.columns:
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    change_features = calculate_change_features(df)
    
    enhanced_features = np.concatenate([
        df[['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'acc_mag']].values,
        np.tile(change_features, (len(df), 1))
    ], axis=1)
    
    return enhanced_features, change_features, df

def is_still(window_df: pd.DataFrame) -> bool:
    """
    Checks if the sensor data in a window indicates stillness.
    Based on low standard deviation of acceleration and low maximum gyroscope values.
    """
    acc_mag = np.sqrt(window_df['ax']**2 + window_df['ay']**2 + window_df['az']**2)
    acc_std = acc_mag.std()
    max_gyro_window = max(window_df['wx'].abs().max(), 
                          window_df['wy'].abs().max(), 
                          window_df['wz'].abs().max())
    
    return bool(acc_std < STILLNESS_ACC_STD_THRESHOLD and max_gyro_window < STILLNESS_GYRO_MAX_THRESHOLD)


@app.post("/predict/")
def predict(data: FallDetectionData):
    """
    Enhanced fall detection with anti-overfitting model and refined change pattern analysis.
    """
    try:
        if model is None:
            return {"error": "Model not loaded", "status": "failed"}
        
        if len(data.sensor_data) != 100:
            return {"error": f"Invalid input length. Expected 100 sensor readings, got {len(data.sensor_data)}", "status": "failed"}
        
        device_id = data.sensor_data[0].device_id

        features, change_features_array, raw_df_window = preprocess_enhanced_sensor_data(data.sensor_data)
        
        expected_features = 17
        if features.shape[1] != expected_features:
            return {"error": f"Invalid feature shape after preprocessing. Expected (100, {expected_features}), got {features.shape}", "status": "failed"}
        
        features = np.expand_dims(features, axis=0)
        
        prediction = model.predict(features)
        
        falling_prob = float(prediction[0][0])
        kneeling_prob = float(prediction[0][1])
        walking_prob = float(prediction[0][2])
        
        max_acc = float(change_features_array[0][0])
        max_change = float(change_features_array[0][1])
        acc_std = float(change_features_array[0][2])
        max_gyro = float(change_features_array[0][3])
        min_acc = float(change_features_array[0][4])
        impact_score = float(change_features_array[0][5])
        ax_change = float(change_features_array[0][6])
        ay_change = float(change_features_array[0][7])
        az_change = float(change_features_array[0][8])
        fall_score = float(change_features_array[0][-1])
        
        # --- Refined Hybrid Decision Logic (Adjusted Thresholds) ---
        
        predicted_class = "unknown"
        confidence = 0.0
        decision_reason = "ML classification with hybrid logic"

        # 1. Strongest physical evidence for a fall (highest thresholds)
        if fall_score > OVERRIDE_FALL_SCORE_HIGH and max_acc > OVERRIDE_MAX_ACC_EXTREME:
            predicted_class = "falling"
            confidence = min(0.98, falling_prob + (fall_score * 0.5))
            decision_reason = "OVERRIDE: Extreme fall pattern & high impact detected"
        # 2. Free-fall + massive impact (also very strong evidence)
        elif max_acc > OVERRIDE_MAX_ACC_EXTREME and min_acc < OVERRIDE_MIN_ACC_EXTREME:
            predicted_class = "falling"
            confidence = min(0.95, falling_prob + 0.4)
            decision_reason = "OVERRIDE: Massive impact & free fall detected"
        # 3. ML is very confident about fall
        elif falling_prob > ML_CONF_FALL_HIGH:
            predicted_class = "falling"
            confidence = falling_prob
            decision_reason = f"ML very confident about fall ({falling_prob:.3f})"
        # 4. Moderate ML confidence + combined sudden change OR free-fall OR high rotation
        # ADJUSTED: Requires higher max_change OR combines max_change with max_gyro
        elif falling_prob > ML_CONF_FALL_MEDIUM and (
            max_change > THRESHOLD_MAX_CHANGE_AMBIGUOUS or # Very high sudden change
            (max_change > THRESHOLD_MAX_CHANGE_ROTATION_COMBINED and max_gyro > THRESHOLD_MAX_GYRO_FALL_SCORE) or # Sudden change + significant rotation
            min_acc < THRESHOLD_MIN_ACC_AMBIGUOUS
        ):
            predicted_class = "falling"
            confidence = min(0.85, falling_prob + 0.3)
            decision_reason = f"ML ({falling_prob:.3f}) + strong change/rotation/free-fall"
        # 5. Moderate ML confidence combined with high impact
        elif falling_prob > ML_CONF_FALL_LOW and impact_score > THRESHOLD_IMPACT_AMBIGUOUS:
            predicted_class = "falling"
            confidence = min(0.80, falling_prob + 0.35)
            decision_reason = f"ML ({falling_prob:.3f}) + high impact"
        # 6. Default to ML model's top choice if no strong fall rules are met
        else:
            predicted_class_index = prediction.argmax(axis=1)[0]
            ml_top_choice = classes[predicted_class_index]
            ml_confidence = float(prediction[0][predicted_class_index])
            
            if ml_top_choice == "falling" and ml_confidence < ML_CONF_FALL_LOW:
                if kneeling_prob > walking_prob:
                    predicted_class = "kneeling"
                    confidence = kneeling_prob
                    decision_reason = f"ML low confidence falling ({ml_confidence:.3f}), re-classified to kneeling"
                else:
                    predicted_class = "walking"
                    confidence = walking_prob
                    decision_reason = f"ML low confidence falling ({ml_confidence:.3f}), re-classified to walking"
            else:
                predicted_class = ml_top_choice
                confidence = ml_confidence
                decision_reason = f"ML classification: {ml_top_choice} (confidence: {ml_confidence:.3f})"
        
        # --- Post-Prediction Confirmation Logic (Temporal Consistency) ---
        
        current_state = fall_prediction_history.get(device_id, {'last_pred': None, 'count': 0})
        
        final_predicted_class = predicted_class
        final_confidence = min(confidence, 1.0)
        final_decision_reason = decision_reason

        # Check for stillness BEFORE a potential fall (to address "not moving gives falling")
        if is_still(raw_df_window) and predicted_class == "falling":
            # If the device is now still AND it was just predicted as 'falling',
            # it's a potential false alarm if the 'fall' didn't have strong characteristics.

            # Re-evaluate the "fall" characteristics that triggered this prediction.
            # If it's a fall prediction that lacks strong impact AND high acceleration,
            # it's likely a controlled placement or gentle drop, not a true fall.
            # True falls often have high impacts OR extreme acceleration (even if short freefall).

            # Check if the fall was NOT due to extreme impact/acceleration overrides
            # (which would indicate a definite fall onto a still surface, e.g., fainting)
            is_fall_from_strong_override = (
                bool(fall_score > OVERRIDE_FALL_SCORE_HIGH and max_acc > OVERRIDE_MAX_ACC_EXTREME) or
                bool(max_acc > OVERRIDE_MAX_ACC_EXTREME and min_acc < OVERRIDE_MIN_ACC_EXTREME)
            )
            
            # Check if the fall was NOT accompanied by a significant impact score or high overall acceleration.
            # This is key for distinguishing controlled placement from a fall.
            is_fall_low_impact_or_acc = (
                bool(impact_score < THRESHOLD_IMPACT_AMBIGUOUS) and # Not a strong impact
                bool(max_acc < OVERRIDE_MAX_ACC_EXTREME) # Not extremely high acceleration (even if not strong override)
            )

            # Rule for overriding fall to "not_a_fall_stillness":
            # If the current window is still, and a 'falling' prediction was made,
            # AND that 'falling' prediction was *not* due to an extremely strong fall override,
            # AND the underlying features (impact/max_acc) suggest a controlled placement,
            # then override it.
            if not is_fall_from_strong_override and is_fall_low_impact_or_acc:
                final_predicted_class = "not_a_fall_controlled_placement" # More specific label
                final_confidence = 0.99
                final_decision_reason = "OVERRIDE: Fall detected during controlled placement onto flat surface."
            # else:
                # If it's still AND predicted as falling, AND had strong impact/accel,
                # then it might be a genuine fall ending in stillness (e.g., fainting, or falling and remaining motionless).
                # The temporal confirmation will then play its role.


        # Update history for this device
        if final_predicted_class == "falling":
            if current_state['last_pred'] == "falling":
                current_state['count'] += 1
            else:
                current_state['count'] = 1
            current_state['last_pred'] = "falling"
        else:
            current_state['count'] = 0
            current_state['last_pred'] = final_predicted_class
        
        fall_prediction_history[device_id] = current_state

        # Apply confirmation threshold
        if current_state['last_pred'] == "falling" and current_state['count'] < CONFIRMATION_WINDOWS:
            final_predicted_class = "pending_fall_confirmation"
            final_decision_reason = f"Fall pending confirmation ({current_state['count']}/{CONFIRMATION_WINDOWS} consecutive)"
            final_confidence = 0.5


        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        # Calculate fall indicators - now with bool() casts
        fall_indicators = {
            "high_impact": bool(max_acc > THRESHOLD_MAX_ACC_FALL_SCORE),
            "free_fall": bool(min_acc < THRESHOLD_MIN_ACC_FALL_SCORE),
            "sudden_change": bool(max_change > THRESHOLD_ACC_DIFF_FALL_SCORE),
            "high_rotation": bool(max_gyro > THRESHOLD_MAX_GYRO_FALL_SCORE),
            "strong_fall_pattern": bool(fall_score > OVERRIDE_FALL_SCORE_HIGH),
            "extreme_acceleration_override": bool(max_acc > OVERRIDE_MAX_ACC_EXTREME),
            "impact_detected_rule": bool(impact_score > THRESHOLD_IMPACT_AMBIGUOUS)
        }
        
        return {
            "device_id": device_id,
            "predicted_class": final_predicted_class,
            "confidence": min(final_confidence, 1.0),
            "decision_reason": final_decision_reason,
            "ml_raw_predictions": prediction_probs,
            "change_analysis_features": {
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
                    "az_max_change": round(az_change, 3)
                },
                 "is_current_window_still": bool(is_still(raw_df_window))
            },
            "fall_indicators": fall_indicators,
            "temporal_confirmation_state": current_state,
            "model_info": {
                "model_type": "Anti-overfitting Enhanced GRU",
                "features_used": expected_features,
                "training_accuracy": "66.6% (realistic, not overfitted)"
            },
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/predict_raw/")
def predict_raw(data: FallDetectionRawData):
    """
    Alternative endpoint for already preprocessed enhanced data.
    Expects 100 time steps with 17 features each (7 original + 10 change detection).
    NOTE: This endpoint does NOT perform full hybrid logic or post-processing,
    as it assumes data is already "raw" from preprocessing.
    """
    try:
        if model is None:
            return {"error": "Model not loaded", "status": "failed"}
        
        # Convert the input list into a numpy array
        features = np.array(data.features)
        
        # Validate input shape (100 time steps, 17 features)
        expected_features = 17
        if features.shape != (100, expected_features):
            return {"error": f"Invalid input shape. Expected (100, {expected_features}), got {features.shape}", "status": "failed"}
        
        # Reshape input to match the model's expected shape
        features = np.expand_dims(features, axis=0)
        
        # Get model prediction
        prediction = model.predict(features)
        
        # Simple classification for raw endpoint
        predicted_class_index = prediction.argmax(axis=1)[0]
        predicted_class = classes[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        prediction_probs = {
            classes[i]: float(prediction[0][i]) 
            for i in range(len(classes))
        }
        
        return {
            "device_id": data.device_id,
            "predicted_class": predicted_class,
            "confidence": min(confidence, 1.0),
            "ml_predictions": prediction_probs,
            "note": "Raw data mode - full hybrid logic and change analysis not available. Only ML output.",
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Enhanced Anti-Overfitting Fall Detection API!",
        "model_loaded": model is not None,
        "model_info": {
            "type": "Anti-overfitting Enhanced GRU with Change Detection",
            "training_accuracy": "66.6% (realistic, well-generalized)",
            "features_per_timestep": 17,
            "overfitting_prevention": "✓ Applied"
        },
        "features": [
            "✓ Anti-overfitting training (66.6% realistic accuracy)",
            "✓ ML-based classification with Bidirectional GRU",
            "✓ Drastic change detection (acceleration spikes)",
            "✓ Free fall pattern recognition (low acceleration)",
            "✓ Impact detection (sudden acceleration jumps)",
            "✓ High rotation detection (gyroscope analysis)",
            "✓ Combined ML + Refined Rule-based Enhancement",
            "✓ Multiple fall indicators",
            "✓ Detailed change analysis",
            "✓ Decision reasoning explanation",
            "✓ Temporal Confirmation to reduce False Alarms",
            "✓ Stillness Detection to filter false alarms"
        ],
        "classes": classes,
        "endpoints": {
            "/predict/": "Send 100 sensor readings with automatic feature enhancement, hybrid logic, and temporal confirmation",
            "/predict_raw/": "Send already preprocessed enhanced data (17 features per timestep) - simpler ML output only",
            "/health": "Check API and model status",
            "/model_info": "Get detailed model information"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "Enhanced Anti-Overfitting Fall Detection",
        "expected_performance": "Much better real-world accuracy due to anti-overfitting training and refined hybrid logic"
    }

@app.get("/model_info")
def model_info():
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        return {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "classes": classes,
            "features": {
                "original_features": ["ax", "ay", "az", "wx", "wy", "wz", "acc_mag"],
                "change_detection_features": [
                    "max_acceleration",
                    "max_change_rate", 
                    "acceleration_std",
                    "max_gyro_velocity",
                    "min_acceleration",
                    "impact_score",
                    "ax_change",
                    "ay_change", 
                    "az_change",
                    "fall_pattern_score"
                ],
                "total_features_per_timestep": 17
            },
            "model_improvements": {
                "anti_overfitting": "✓ Aggressive regularization applied",
                "early_stopping": "✓ Stopped at epoch 21 (instead of 50)",
                "realistic_accuracy": "66.6% (much more reliable than 99%)",
                "change_detection": "✓ Physics-based fall pattern recognition",
                "combined_approach": "✓ ML + rule-based decision making"
            },
            "fall_detection_thresholds_current_config": {
                "rule_based_feature_calc": {
                    "max_acc_fall_score": f"> {THRESHOLD_MAX_ACC_FALL_SCORE} m/s²",
                    "min_acc_fall_score": f"< {THRESHOLD_MIN_ACC_FALL_SCORE} m/s²",
                    "acc_diff_fall_score": f"> {THRESHOLD_ACC_DIFF_FALL_SCORE} m/s²",
                    "max_gyro_fall_score": f"> {THRESHOLD_MAX_GYRO_FALL_SCORE} rad/s",
                    "impact_pre_acc": f"< {THRESHOLD_IMPACT_PRE_ACC} m/s²",
                    "impact_post_acc": f"> {THRESHOLD_IMPACT_POST_ACC} m/s²",
                },
                "hybrid_decision_logic_overrides": {
                    "override_fall_score_high": f"> {OVERRIDE_FALL_SCORE_HIGH}",
                    "override_max_acc_extreme": f"> {OVERRIDE_MAX_ACC_EXTREME} m/s²",
                    "override_min_acc_extreme": f"< {OVERRIDE_MIN_ACC_EXTREME} m/s²",
                    "ml_conf_fall_high": f"> {ML_CONF_FALL_HIGH}",
                    "ml_conf_fall_medium": f"> {ML_CONF_FALL_MEDIUM}",
                    "ml_conf_fall_low": f"> {ML_CONF_FALL_LOW}",
                    "threshold_max_change_ambiguous": f"> {THRESHOLD_MAX_CHANGE_AMBIGUOUS} m/s²",
                    "threshold_min_acc_ambiguous": f"< {THRESHOLD_MIN_ACC_AMBIGUOUS} m/s²",
                    "threshold_impact_ambiguous": f"> {THRESHOLD_IMPACT_AMBIGUOUS}",
                    "threshold_max_change_rotation_combined": f"> {THRESHOLD_MAX_CHANGE_ROTATION_COMBINED} m/s² (combined with max_gyro)"
                },
                "post_prediction_confirmation": {
                    "confirmation_windows": f"{CONFIRMATION_WINDOWS} consecutive 'falling' predictions",
                    "stillness_acc_std_threshold": f"< {STILLNESS_ACC_STD_THRESHOLD} std dev",
                    "stillness_gyro_max_threshold": f"< {STILLNESS_GYRO_MAX_THRESHOLD} max rad/s",
                    "stillness_duration_windows": f"{STILLNESS_DURATION_WINDOWS} windows"
                }
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Enhanced Anti-Overfitting Fall Detection API...")
    print("Model features:")
    print("- ✓ Realistic 66.6% accuracy (not overfitted)")
    print("- ✓ Change detection for drastic movements")
    print("- ✓ Physics-based fall pattern recognition")
    print("- ✓ Combined ML + rule-based decisions (now refined)")
    print("- ✓ Detailed analysis and reasoning")
    print("- ✓ Added Temporal Confirmation and Stillness Detection for false alarm reduction.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
