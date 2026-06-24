"""
DataPipeline_Fixed.py

Builds the training dataset for a tendon/servo-driven robotic hand.

Goal:
    raw camera/gesture data  ->  clean supervised dataset

Expected raw CSV format, either:

1) Already-computed finger bend columns:
    thumb,index,middle,ring,pinky,servo1,servo2,servo3,servo4,servo5

2) MediaPipe-style landmark columns:
    x0,y0,z0,x1,y1,z1,...,x20,y20,z20,servo1,...,servo5

Output:
    data/processed/hand_dataset.npz

Dataset meaning:
    finger_pose = [thumb,index,middle,ring,pinky] in [0,1]
    servo_angles = [servo1,servo2,servo3,servo4,servo5] in degrees

This is the hand version of the Flexi-Robot DataPipeline idea:
    soft-arm actuation/pose dataset -> hand servo/finger-pose dataset
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw_hand_log.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "hand_dataset.npz"

FINGER_COLS = ["thumb", "index", "middle", "ring", "pinky"]
SERVO_COLS = ["servo1", "servo2", "servo3", "servo4", "servo5"]

# MediaPipe landmark indices for each finger.
# We estimate bend from three joints per finger.
FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in radians."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cosang = float(np.dot(ba, bc) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang)


def normalize_bend(angle_rad: float) -> float:
    """
    Convert joint angle to bend score in [0,1].
    Open finger is close to pi radians; bent finger is smaller.
    """
    score = (math.pi - angle_rad) / math.pi
    return float(max(0.0, min(1.0, score)))


def compute_finger_bends_from_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute thumb/index/middle/ring/pinky bend scores from x0,y0,z0...x20,y20,z20."""
    out = pd.DataFrame(index=df.index)

    for finger, joints in FINGER_JOINTS.items():
        values = []
        for _, row in df.iterrows():
            pts = []
            for j in joints:
                pts.append(np.array([row[f"x{j}"], row[f"y{j}"], row.get(f"z{j}", 0.0)], dtype=float))
            values.append(normalize_bend(angle_between(pts[0], pts[1], pts[2])))
        out[finger] = values

    return out


def has_landmarks(df: pd.DataFrame) -> bool:
    needed = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
    return all(c in df.columns for c in needed)


def make_synthetic_dataset(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Fallback dataset for testing when no real CSV has been recorded yet."""
    rng = np.random.default_rng(42)
    finger_pose = rng.uniform(0, 1, size=(n, 5))

    # Simple initial mapping: more bend -> larger servo angle.
    servo_min = np.array([20, 20, 20, 20, 20], dtype=float)
    servo_max = np.array([140, 140, 140, 140, 140], dtype=float)
    servo_angles = servo_min + finger_pose * (servo_max - servo_min)
    servo_angles += rng.normal(0, 2.0, size=servo_angles.shape)

    return finger_pose.astype(np.float32), servo_angles.astype(np.float32)


def load_or_build_dataset(raw_path: Path = RAW_PATH) -> tuple[np.ndarray, np.ndarray]:
    if not raw_path.exists():
        print(f"[WARN] {raw_path} not found. Creating synthetic test dataset.")
        return make_synthetic_dataset()

    df = pd.read_csv(raw_path)

    if all(c in df.columns for c in FINGER_COLS):
        finger_df = df[FINGER_COLS].copy()
    elif has_landmarks(df):
        finger_df = compute_finger_bends_from_landmarks(df)
    else:
        raise ValueError(
            "CSV must contain either finger bend columns "
            f"{FINGER_COLS} or MediaPipe landmarks x0,y0,z0...x20,y20,z20."
        )

    if not all(c in df.columns for c in SERVO_COLS):
        raise ValueError(f"CSV must contain servo columns: {SERVO_COLS}")

    servo_df = df[SERVO_COLS].copy()

    dataset = pd.concat([finger_df, servo_df], axis=1)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    finger_pose = dataset[FINGER_COLS].to_numpy(dtype=np.float32)
    servo_angles = dataset[SERVO_COLS].to_numpy(dtype=np.float32)

    finger_pose = np.clip(finger_pose, 0.0, 1.0)
    servo_angles = np.clip(servo_angles, 0.0, 180.0)

    return finger_pose, servo_angles


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    finger_pose, servo_angles = load_or_build_dataset()

    np.savez(
        OUT_PATH,
        finger_pose=finger_pose,
        servo_angles=servo_angles,
        finger_cols=np.array(FINGER_COLS),
        servo_cols=np.array(SERVO_COLS),
    )

    print("[OK] Dataset saved:", OUT_PATH)
    print("finger_pose shape:", finger_pose.shape)
    print("servo_angles shape:", servo_angles.shape)


if __name__ == "__main__":
    main()
