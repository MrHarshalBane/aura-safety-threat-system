# Aura AI — Women's Safety & Threat Detection System

[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-blue?style=flat&logo=github)](https://mrharshalbane.github.io/aura-safety-threat-system/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)

An AI/ML-driven women's safety platform built for **SIH 2026** by Team SheShield (Team #20, TCET). Secured **8th place out of 20 teams** in the AI/ML track.

---

## 🌐 Live Web Prototype
Access the instant in-browser prototype running on GitHub Pages:
👉 **[https://mrharshalbane.github.io/aura-safety-threat-system/](https://mrharshalbane.github.io/aura-safety-threat-system/)**

- **In-Browser Camera Detection**: Prompts for camera permission to perform live pose estimation and distress signal detection right inside the browser.
- **SOS Gesture Detection**: Automatically triggers emergency escalation when raised hands or surrender/distress gestures are sustained.
- **Telegram Emergency Dispatch**: Connects directly to Telegram Bot API to dispatch emergency alerts with location, trigger reason, and timestamp.

---

## 🚀 Running the Python Streamlit & YOLOv8 Engine Locally

The Python engine integrates **YOLOv8-pose**, **OpenCV DNN Caffe gender classification**, **SQLite incident logging**, and **asynchronous Telegram dispatching**.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Telegram Alerts (Optional)
Set your environment variables or enter them directly inside the Streamlit sidebar:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 🛡️ Core Capabilities

1. **SOS Gesture Detection**: Real-time skeletal tracking monitors wrist and shoulder keypoints to detect hands-up distress signals within ~0.15s.
2. **Night Vigilance Protocol**: Dynamically increases vigilance during night hours (20:00 to 06:00), detecting isolated individuals and potential stalking or surrounding patterns.
3. **Decoupled 30+ FPS Inference Worker**: Background thread decouples video capture from YOLOv8-pose inference, guaranteeing high-framerate camera feed.
4. **Instant Telegram Cloud Dispatch**: High-priority alert payload dispatched asynchronously to security channels.
5. **Persistent Incident Database**: SQLite storage logs each threat state transition, person counts, and dispatch IDs.
