"""
Train_InverseNN_v2.py

Inverse model for robotic hand:
    finger_pose -> servo_angles

Meaning:
    Given target finger bends, predict 5 servo angles.

This is the model you will eventually call from:
    camera hand pose  ->  target finger_pose  ->  servo_angles
    EEG OPEN/CLOSE    ->  target finger_pose  ->  servo_angles
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
MODEL_PATH = MODEL_DIR / "inverse_pose_to_servo.joblib"
METRICS_PATH = MODEL_DIR / "inverse_metrics.json"


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python hand_ai_pipeline/DataPipeline_Fixed.py"
        )

    data = np.load(DATA_PATH, allow_pickle=True)
    finger_pose = data["finger_pose"].astype(np.float32)
    servo_angles = data["servo_angles"].astype(np.float32)
    return finger_pose, servo_angles


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
    pred = np.clip(model.predict(X_test), 0.0, 180.0)

    metrics = {
        "task": "inverse_pose_to_servo",
        "input": "thumb,index,middle,ring,pinky bend in [0,1]",
        "output": "servo1..servo5 in degrees",
        "samples_total": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "mae_degrees": float(mean_absolute_error(y_test, pred)),
        "rmse_degrees": float(np.sqrt(mean_squared_error(y_test, pred))),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[OK] Inverse model saved:", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


def demo_predict_open_close() -> None:
    """Tiny sanity check after training."""
    model = joblib.load(MODEL_PATH)

    open_pose = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    close_pose = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    print("OPEN servo:", np.clip(model.predict(open_pose)[0], 0, 180).round(1))
    print("CLOSE servo:", np.clip(model.predict(close_pose)[0], 0, 180).round(1))


if __name__ == "__main__":
    train()
    demo_predict_open_close()
