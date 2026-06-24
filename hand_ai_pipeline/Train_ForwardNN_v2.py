"""
Train_ForwardNN_v2.py

Forward model for robotic hand:
    servo_angles -> finger_pose

Meaning:
    Given 5 servo angles, predict 5 finger bend values.

This is useful for calibration:
    If the mechanical hand does not move exactly as expected,
    the forward model learns the real relationship between servo command and finger pose.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path("data/processed/hand_dataset.npz")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "forward_servo_to_pose.joblib"
METRICS_PATH = MODEL_DIR / "forward_metrics.json"


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python hand_ai_pipeline/DataPipeline_Fixed.py"
        )

    data = np.load(DATA_PATH, allow_pickle=True)
    servo_angles = data["servo_angles"].astype(np.float32)
    finger_pose = data["finger_pose"].astype(np.float32)
    return servo_angles, finger_pose


def train() -> None:
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=2000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0.0, 1.0)

    metrics = {
        "task": "forward_servo_to_pose",
        "input": "servo1..servo5 in degrees",
        "output": "thumb,index,middle,ring,pinky bend in [0,1]",
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "mae_bend": float(mean_absolute_error(y_test, pred)),
        "rmse_bend": float(np.sqrt(mean_squared_error(y_test, pred))),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[OK] Forward model saved:", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
