# Hand AI Pipeline

This folder is the robotic-hand version of the Flexi-Robot modeling idea.

Instead of modeling a continuum soft arm:

```text
act1, act2 -> x, y, z, qx, qy, qz, qw
```

we model a tendon/servo-driven hand:

```text
servo1..servo5 -> thumb,index,middle,ring,pinky
thumb,index,middle,ring,pinky -> servo1..servo5
```

## Files

```text
DataPipeline_Fixed.py
```

Builds a clean dataset from either:

1. Direct finger-bend CSV:

```csv
thumb,index,middle,ring,pinky,servo1,servo2,servo3,servo4,servo5
0.0,0.1,0.0,0.0,0.2,20,30,20,20,40
```

2. MediaPipe landmark CSV:

```csv
x0,y0,z0,x1,y1,z1,...,x20,y20,z20,servo1,servo2,servo3,servo4,servo5
```

Output:

```text
data/processed/hand_dataset.npz
```

---

```text
Train_ForwardNN_v2.py
```

Forward model:

```text
servo_angles -> finger_pose
```

Useful for learning how the real printed hand actually moves.

---

```text
Train_InverseNN_v2.py
```

Inverse model:

```text
finger_pose -> servo_angles
```

This is the model used by camera control and later EEG high-level commands.

## Run

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

Build dataset:

```bash
python hand_ai_pipeline/DataPipeline_Fixed.py
```

Train forward model:

```bash
python hand_ai_pipeline/Train_ForwardNN_v2.py
```

Train inverse model:

```bash
python hand_ai_pipeline/Train_InverseNN_v2.py
```

## Current design

For the first demo, use only high-level poses:

```text
OPEN  = [0,0,0,0,0]
CLOSE = [1,1,1,1,1]
PINCH = [1,1,0,0,0]
```

Later EEG should output only high-level intent:

```text
EEG -> OPEN / CLOSE / HOLD
```

Then the inverse model converts that intent pose into servo angles.
