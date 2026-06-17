# Snydor — EEG + 腱绳仿生机械手 BCI 系统

## 🧠 项目简介
Snydor 是一个基于 EEG（脑电信号）的脑机接口（BCI）控制系统，用于驱动腱绳仿生机械手，实现运动想象控制（Motor Imagery Control）。

系统目标：
- EEG 采集（8通道优先，支持现成数据对照）
- ICA 去伪迹 + CSP 特征提取
- SVM / ML 分类运动想象
- ESP32 实时控制舵机
- 腱绳驱动仿生机械手（可扩展五指独立控制）

---

## 🧩 系统架构

EEG采集 → Python信号处理 → ICA去伪迹 → CSP特征 → 分类器 → ESP32 → PCA9685 → 舵机 → 腱绳机械手

---

## 🦾 机械结构（Snydor Hand）

- 腱绳驱动（tendon-driven）
- 背侧弹性回弹（橡皮筋）
- 前臂舵机远端驱动
- v1：双舵机控制（四指联动 + 拇指）
- v2：五舵机独立控制预留

关键参数：
- 手掌：90×75×15 mm
- 前臂底座：150×95×50 mm
- 腱绳直径：1.2–1.5 mm
- 铰链：M2 / 2mm轴

---

## 🧠 EEG Pipeline

1. 信号采集（OpenBCI / ADS1299 / 现成数据）
2. 带通滤波（1–40Hz）
3. ICA 去伪迹（眼电/肌电）
4. CSP 特征提取
5. SVM / LDA 分类

分类任务：
- Rest
- Right-hand motor imagery
- Artifact（可选扩展）

---

## ⚙️ 硬件组成

- ESP32（通信与控制）
- PCA9685（16路PWM）
- 舵机（MG996R / DS3218）
- 3D打印腱绳机械手

通信方式：
- Serial / WiFi UDP

---

## 📁 项目结构（规划）

```
eeg/              # EEG采集 + ICA + CSP
control/          # ESP32通信控制
firmware/         # ESP32代码
mechanical_hand/  # CAD + 腱绳结构
experiment/       # 运动想象实验
models/           # 训练模型
docs/             # 技术文档
```

---

## 🚀 当前开发阶段

当前目标（Phase 1）：
- EEG 数据采集跑通
- ICA + CSP pipeline
- 两类分类（Rest vs MI）
- 控制机械手 OPEN / GRASP

---

## 🎯 项目理念

本项目不以论文为目标，而以：
- 可运行系统
- 可复现流程
- GitHub完整记录
- 视频展示成果

为核心。

---

## 📌 下一步

1. EEG采集脚本（Python）
2. ESP32通信协议
3. 腱绳机械手CAD建模
4. 训练 baseline classifier
