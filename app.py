import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import queue
import time
import os
import tempfile
import torch
import altair as alt
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

warnings.filterwarnings("ignore")

# Local Aura AI Modules
import database
from threat_engine import ThreatAnalysisEngine
from telegram_alerts import get_telegram_credentials, dispatch_telegram_alert_async, send_telegram_message_sync, safe_log
from gender_classifier import GenderClassifier

# ==========================================
# 1. DATABASE & THREAT ENGINE INIT
# ==========================================
database.init_db()
# Instant 2-frame trigger (~0.15s) with 4-frame flicker tolerance
threat_engine = ThreatAnalysisEngine(required_sos_frames=2, max_missed_frames=4)

# Hardware Device Detection & Performance Tuning
if torch.cuda.is_available():
    INFERENCE_DEVICE = "cuda"
else:
    INFERENCE_DEVICE = "cpu"
    try:
        torch.set_num_threads(4)
    except Exception:
        pass

# ==========================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Aura - Women's Safety AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 3. HIGH-CONTRAST BENTO-GRID CSS
# ==========================================
st.markdown(
    """
    <style>
    /* -----------------------------------------
       1. FONTS & GLOBAL LIGHT THEME
    ----------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F3F4F6 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #111827 !important;
    }

    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #111827 !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #111827 !important;
    }

    /* -----------------------------------------
       2. SIDEBAR CONTAINER & CONTRAST
    ----------------------------------------- */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #111827 !important;
    }

    /* High-contrast Sidebar Toggle */
    [data-testid="stToggle"] {
        background-color: #F9FAFB !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        border: 1px solid #D1D5DB !important;
    }
    [data-testid="stToggle"] label p {
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* Inputs, Sliders, and Tooltips */
    input, input[type="number"], div[data-baseweb="input"] {
        color: #111827 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
    }
    [data-testid="stSlider"] div, [data-testid="stSlider"] span {
        color: #111827 !important;
        font-weight: 500 !important;
    }
    [data-testid="stTooltipIcon"] svg {
        fill: #4B5563 !important;
        stroke: #4B5563 !important;
    }

    /* -----------------------------------------
       3. BENTO-GRID PANELS & CARDS
    ----------------------------------------- */
    .aura-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
        border: 1px solid #E5E7EB;
    }
    .aura-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .aura-badge {
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .aura-mission {
        font-size: 0.86rem;
        color: #4B5563;
        font-weight: 500;
        margin-top: 4px;
    }

    .bento-card {
        background: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        padding: 18px;
        margin-bottom: 16px;
    }
    .bento-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #4B5563;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 3-Tier Threat Indicators */
    .threat-status-box {
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
    }
    .status-safe {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46 !important;
    }
    .status-caution {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E !important;
    }
    .status-alert {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B !important;
        animation: pulseAlert 1.4s infinite ease-in-out;
    }
    @keyframes pulseAlert {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .metric-value-large {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .metric-subtitle {
        font-size: 0.78rem;
        color: #6B7280;
    }

    /* Activity Log Box */
    .log-scroll-area {
        max-height: 190px;
        overflow-y: auto;
        padding-right: 4px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .log-scroll-area::-webkit-scrollbar { width: 4px; }
    .log-scroll-area::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 4px; }

    .log-entry {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 10px;
        border-radius: 8px;
        font-size: 0.76rem;
        background: #F9FAFB;
        border: 1px solid #F3F4F6;
    }
    .log-entry.alert { background: #FEF2F2; border-color: #FEE2E2; color: #991B1B !important; font-weight: 600; }
    .log-entry.caution { background: #FFFBEB; border-color: #FEF3C7; color: #B45309 !important; font-weight: 600; }
    .log-entry.safe { background: #F0FDF4; border-color: #DCFCE7; color: #166534 !important; }
    .log-time { color: #9CA3AF !important; font-family: monospace; font-size: 0.7rem; }

    /* Video Live Tag & Pulsing Dot */
    .video-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F3F4F6;
    }
    .live-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #EF4444 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pulse-dot {
        width: 8px; height: 8px;
        background-color: #EF4444;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulseDot 1.4s infinite;
    }
    @keyframes pulseDot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 4. TOP NAVIGATION & MISSION HEADER
# ==========================================
st.markdown(
    """
    <div class="aura-header">
        <div>
            <div class="aura-title">
                <span>🛡️</span>
                <span>Aura <strong>Vision Engine</strong></span>
                <span class="aura-badge">v2.6 Gender-Aware AI</span>
            </div>
            <div class="aura-mission">
                Autonomous Vulnerability Detection, Lone-Woman Night Protocol & Rapid Security Dispatch
            </div>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.8rem; font-weight: 600; color: #4F46E5; background: #EEF2FF; padding: 6px 12px; border-radius: 8px;">
                ⚡ Multi-Signal Fusion + Caffe DNN
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 5. SIDEBAR CONFIGURATION & SETTINGS
# ==========================================
st.sidebar.markdown("### ⚙️ Video & AI Controls")

video_source_mode = st.sidebar.radio(
    "Select Video Source",
    ["Browser WebRTC (Cloud Ready)", "Threaded Local Camera (Ultra-Fast)", "Upload Video File (.mp4)"],
    index=1,
    help="Threaded Local Camera gives 30+ FPS decoupled performance. WebRTC enables browser streaming on cloud deployments."
)

run_stream = st.sidebar.toggle("📹 Camera Active", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ AI Sensitivity & Protocol")

conf_threshold = st.sidebar.slider(
    "Person Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Confidence threshold for YOLOv8-pose person detection."
)

simulated_hour = st.sidebar.slider(
    "Simulated Time of Day (Hour)",
    min_value=0,
    max_value=23,
    value=datetime.now().hour,
    step=1,
    help="Hours between 20:00 (8 PM) and 06:00 (6 AM) activate elevated Night Sensitivity Protocol (including Lone Woman at Night detection)."
)

cam_index = st.sidebar.number_input(
    "Device Index (for local cam)",
    min_value=0,
    max_value=5,
    value=0,
    step=1
)

st.sidebar.caption(
    "ℹ️ **Demographic & Model Note**: Gender classification is a probabilistic visual estimate with known accuracy limitations. "
    "It is utilized strictly as a supporting risk signal in multi-signal fusion (e.g. lone-woman night vigilance), not as a definitive identity determination."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Telegram Emergency Dispatch")

env_token, env_chat_id = get_telegram_credentials()

user_tg_token = st.sidebar.text_input(
    "Telegram Bot Token",
    value=env_token or "",
    type="password",
    help="Paste your Bot Token from @BotFather (e.g. 123456789:ABCDefGh...)"
).strip()

user_tg_chat_id = st.sidebar.text_input(
    "Telegram Chat ID",
    value=env_chat_id or "",
    help="Paste your personal or channel numeric Chat ID (e.g. 987654321)"
).strip()

if user_tg_token:
    clean_token = user_tg_token[3:].strip() if user_tg_token.lower().startswith("bot") else user_tg_token
    os.environ["TELEGRAM_BOT_TOKEN"] = clean_token
if user_tg_chat_id:
    os.environ["TELEGRAM_CHAT_ID"] = user_tg_chat_id

tg_token, tg_chat_id = get_telegram_credentials()

if tg_token and tg_chat_id:
    st.sidebar.success("📱 Telegram Dispatch: Connected", icon="✅")
    if st.sidebar.button("📨 Send Test Telegram Ping"):
        safe_log("[DEBUG APP UI] 'Send Test Telegram Ping' button clicked. Attempting test send...")
        with st.sidebar:
            success = send_telegram_message_sync(
                "🔔 *Aura System Test*: Telegram Emergency Dispatch channel connected and operational.",
                bot_token=tg_token,
                chat_id=tg_chat_id
            )
            if success:
                st.success("Test ping delivered to Telegram!")
            else:
                st.error("Telegram API call failed. Verify token & chat ID.")
else:
    st.sidebar.warning("⚠️ Enter Bot Token & Chat ID above to enable live Telegram alerts.")

emergency_recipient = st.sidebar.text_input("Emergency Contact Label", value="Campus Security (+1-800-555-AURA)")
enable_audio_alarm = st.sidebar.checkbox("🔊 Enable In-Browser Audio Alarm", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"• Hardware Engine: **{'CUDA GPU 🚀' if INFERENCE_DEVICE == 'cuda' else 'Optimized Multi-Threaded CPU ⚡'}**\n• AI Pipeline: **YOLOv8-Pose + Caffe Gender Classifier**")

# ==========================================
# 6. AI MODEL INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_pose_model():
    """Loads YOLOv8-pose model for simultaneous person detection and skeletal analysis."""
    model = YOLO("yolov8n-pose.pt")
    if INFERENCE_DEVICE == "cuda":
        model.to("cuda")
    return model

@st.cache_resource
def load_gender_model():
    """Loads Caffe DNN model for lightweight per-person gender classification."""
    return GenderClassifier()

with st.spinner("Initializing YOLOv8-Pose & Gender Vision Engine..."):
    model = load_pose_model()
    gender_classifier = load_gender_model()

# ==========================================
# 7. DECOUPLED INFERENCE WORKER CLASS
# ==========================================
class DecoupledInferencePipeline:
    """
    Decouples frame capture/display from AI inference.
    - Background thread continuously processes the newest frame at downscaled resolution.
    - Runs YOLOv8-pose + Caffe gender classification on detected bounding boxes.
    - Display thread reads camera frames at 30+ FPS without blocking on inference.
    - Caches latest bounding boxes, keypoints, gender breakdown, and threat state for instant overlay.
    """
    def __init__(self, model, threat_engine, gender_classifier, conf_thresh: float = 0.35, sim_hour: Optional[int] = None):
        self.model = model
        self.threat_engine = threat_engine
        self.gender_classifier = gender_classifier
        self.conf_thresh = conf_thresh
        self.sim_hour = sim_hour
        
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.lock: threading.Lock = threading.Lock()
        self.running: bool = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Thread-safe cached result state
        self.latest_result: Dict[str, Any] = {
            "threat_level": "SAFE",
            "person_count": 0,
            "gender_breakdown": {"female": 0, "male": 0, "unknown": 0},
            "gender_data": [],
            "summary_reason": "System Initialized — Standby",
            "sos_detected": False,
            "sos_progress": 0.0,
            "prox_alert": False,
            "motion_alert": False,
            "is_night": False,
            "time_factor_text": "Standard Protocol",
            "dispatch_payload": None,
            "boxes": np.array([]),
            "keypoints_data": None
        }

    def start(self) -> None:
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submits frame to inference worker, dropping stale frames to prevent backlog."""
        if not self.running:
            return
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _inference_loop(self) -> None:
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            orig_h, orig_w = frame.shape[:2]

            # 1. Downscale frame before inference (480p width) for high FPS
            infer_w = 480
            infer_h = int(orig_h * (infer_w / orig_w))
            frame_small = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

            # 2. Run YOLOv8-pose inference
            try:
                results = self.model.predict(
                    frame_small,
                    conf=self.conf_thresh,
                    classes=[0], # Person class
                    device=INFERENCE_DEVICE,
                    imgsz=384,
                    verbose=False
                )
                
                # Scale boxes back to original display dimensions
                scale_x = orig_w / infer_w
                scale_y = orig_h / infer_h
                
                boxes_scaled = []
                if len(results) > 0 and results[0].boxes is not None:
                    raw_boxes = results[0].boxes.xyxy.cpu().numpy()
                    for b in raw_boxes:
                        boxes_scaled.append([b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y])
                boxes_scaled = np.array(boxes_scaled)

                keypoints = results[0].keypoints if len(results) > 0 else None

                # 3. Lightweight Gender Classification per person crop
                gender_data = []
                if len(boxes_scaled) > 0:
                    gender_data = self.gender_classifier.classify_all_persons(
                        frame=frame,
                        boxes=boxes_scaled,
                        min_confidence=0.65
                    )

                # 4. Evaluate Multi-Signal Threat Logic with Gender & Night Analysis
                threat_res = self.threat_engine.evaluate_threat(
                    boxes=boxes_scaled,
                    keypoints=keypoints,
                    frame_shape=(orig_h, orig_w),
                    simulated_hour=self.sim_hour,
                    gender_data=gender_data
                )

                # 5. Atomically update shared state
                with self.lock:
                    self.latest_result = {
                        **threat_res,
                        "boxes": boxes_scaled,
                        "gender_data": gender_data,
                        "raw_results": results[0] if len(results) > 0 else None,
                        "scale_x": scale_x,
                        "scale_y": scale_y
                    }

            except Exception as e:
                safe_log(f"[DEBUG INFERENCE ERROR] Exception in inference loop: {type(e).__name__} -> {e}")

    def get_latest_result(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_result)

    def process_display_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Unified method: Retrieves latest inference result and renders overlays with gender labels on the frame.
        Guarantees that the on-screen banner and dispatch decision read from the EXACT same state object.
        """
        res = self.get_latest_result()
        boxes = res.get("boxes", [])
        gender_data = res.get("gender_data", [])
        threat_level = res.get("threat_level", "SAFE")
        sos_detected = res.get("sos_detected", False)
        sos_progress = res.get("sos_progress", 0.0)

        # Color coding: Green for SAFE, Amber for CAUTION, Red for ALERT
        if threat_level == "ALERT":
            box_color = (0, 0, 235) # Red
        elif threat_level == "CAUTION":
            box_color = (0, 180, 245) # Amber / Yellow
        else:
            box_color = (60, 200, 70) # Green

        # Draw bounding boxes with gender indicators
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Format label with gender & confidence if available
            label_text = "PERSON"
            if i < len(gender_data):
                g_info = gender_data[i]
                g_label = g_info.get("gender", "Unknown")
                g_conf = g_info.get("confidence", 0.0)
                if g_label in ["Female", "Male"]:
                    label_text = f"PERSON: {g_label} {int(g_conf * 100)}%"

            # Label badge with background fill for high contrast
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + label_size[0] + 6, y1), box_color, -1)
            cv2.putText(frame, label_text, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Top Threat Banner
        h, w = frame.shape[:2]
        if threat_level == "ALERT":
            cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 220), -1)
            banner_text = f"CRITICAL ALERT: {res.get('summary_reason', 'Threat Detected')[:50]}"
            cv2.putText(frame, banner_text, (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
        elif threat_level == "CAUTION":
            cv2.rectangle(frame, (0, 0), (w, 38), (0, 140, 220), -1)
            cv2.putText(frame, f"CAUTION: {res.get('summary_reason', 'Elevated Vigilance')[:50]}", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)

        # On-Screen Visual Indicator for Gesture Tracking Progress
        if 0.0 < sos_progress < 1.0:
            bar_box_w = min(480, w - 40)
            x_start = 20
            y_start = h - 60
            cv2.rectangle(frame, (x_start, y_start), (x_start + bar_box_w, y_start + 45), (20, 20, 20), -1)
            cv2.rectangle(frame, (x_start, y_start), (x_start + bar_box_w, y_start + 45), (0, 200, 255), 2)
            
            fill_w = int((bar_box_w - 20) * sos_progress)
            cv2.rectangle(frame, (x_start + 10, y_start + 25), (x_start + 10 + fill_w, y_start + 35), (0, 200, 255), -1)
            
            pct = int(sos_progress * 100)
            cv2.putText(
                frame,
                f"DETECTING SOS GESTURE: {pct}% (Hold Position...)",
                (x_start + 12, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        elif sos_detected:
            cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (0, 0, 200), -1)
            cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (255, 255, 255), 2)
            cv2.putText(
                frame,
                "🚨 CONFIRMED SOS DISTRESS GESTURE ACTIVATED",
                (24, h - 22),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame, res

# ==========================================
# 8. ASYMMETRICAL BENTO GRID LAYOUT
# ==========================================
col_left, col_right = st.columns([7, 3], gap="medium")

# ----------------- LEFT COLUMN: VIDEO FEED -----------------
with col_left:
    st.markdown(
        """
        <div class="bento-card" style="margin-bottom: 0px;">
            <div class="video-panel-header">
                <div style="font-weight: 700; font-size: 1rem; color: #111827;">
                    Live Surveillance Stream
                </div>
                <div class="live-tag">
                    <span class="pulse-dot"></span> LIVE VISION
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    video_placeholder = st.empty()
    dispatch_card_placeholder = st.empty()

# ----------------- RIGHT COLUMN: BENTO METRIC PANELS -----------------
with col_right:
    threat_placeholder = st.empty()
    count_placeholder = st.empty()
    log_placeholder = st.empty()
    chart_placeholder = st.empty()

audio_placeholder = st.empty()

def trigger_audio_alarm() -> None:
    """Generates an immediate in-browser two-tone emergency buzzer using HTML5 Web Audio API."""
    if enable_audio_alarm:
        audio_placeholder.markdown(
            """
            <script>
            try {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = audioCtx.createOscillator();
                var gain = audioCtx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                osc.frequency.setValueAtTime(440, audioCtx.currentTime + 0.15);
                gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.35);
            } catch(e){}
            </script>
            """,
            unsafe_allow_html=True
        )

def render_db_activity_logs() -> str:
    """Renders the recent activity log from SQLite database."""
    incidents = database.get_recent_incidents(limit=10)
    if not incidents:
        return """
        <div class="bento-card">
            <div class="bento-card-title">📜 Incident Activity Log</div>
            <div style="font-size: 0.8rem; color: #6B7280; text-align: center; padding: 16px;">No incidents recorded in database.</div>
        </div>
        """
    log_entries_html = ""
    for item in incidents:
        t_level = item["threat_level"].lower()
        log_entries_html += f"""
        <div class="log-entry {t_level}">
            <span>{item['reason']}</span>
            <span class="log-time">{item['time']}</span>
        </div>
        """
    return f"""
    <div class="bento-card">
        <div class="bento-card-title">📜 Incident Activity Log (SQLite)</div>
        <div class="log-scroll-area">
            {log_entries_html}
        </div>
    </div>
    """

def render_hourly_chart() -> None:
    """Renders Altair hourly alert frequency chart."""
    df = database.get_hourly_incident_stats()
    chart = alt.Chart(df).mark_bar(color="#4F46E5", cornerRadius=4).encode(
        x=alt.X("hour:N", title="Hour of Day", axis=alt.Axis(labelAngle=0, labelColor="#4B5563")),
        y=alt.Y("count:Q", title="Alerts", axis=alt.Axis(tickMinStep=1, labelColor="#4B5563")),
        tooltip=["hour", "count"]
    ).properties(
        height=140
    ).configure_view(
        strokeOpacity=0
    )
    chart_placeholder.markdown("""<div class="bento-card"><div class="bento-card-title">📊 Threat Density Trend (Hourly)</div>""", unsafe_allow_html=True)
    chart_placeholder.altair_chart(chart, use_container_width=True)

# Initial Render of Panels
log_placeholder.markdown(render_db_activity_logs(), unsafe_allow_html=True)
render_hourly_chart()

# ==========================================
# 9. VIDEO CAPTURE & STREAMING PIPELINES
# ==========================================
if run_stream:
    pipeline = DecoupledInferencePipeline(
        model=model,
        threat_engine=threat_engine,
        gender_classifier=gender_classifier,
        conf_thresh=conf_threshold,
        sim_hour=simulated_hour
    )
    pipeline.start()

    last_logged_state: Optional[str] = None
    last_logged_time: float = 0
    last_alert_beep: float = 0
    
    # Telegram Rate Limiting & Cooldown Tracking
    last_telegram_state: Optional[str] = None
    last_telegram_time: float = 0       # 60s cooldown for general crowding/proximity
    last_sos_active: bool = False       # Tracks whether an SOS distress gesture is actively ongoing
    last_sos_telegram_time: float = 0   # Cooldown for repeating sustained SOS alerts

    # ----------------------------------------------------
    # MODE A: THREADED LOCAL WEBCAM (Ultra-Low Latency)
    # ----------------------------------------------------
    if video_source_mode == "Threaded Local Camera (Ultra-Fast)":
        cap = cv2.VideoCapture(int(cam_index))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error(f"❌ Cannot connect to webcam at index {cam_index}.")
            pipeline.stop()
        else:
            try:
                frame_count = 0
                while run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Frame capture interrupted.")
                        break

                    frame_count += 1
                    if frame_count % 2 == 0:
                        pipeline.submit_frame(frame)

                    # Unified overlay processing and state retrieval
                    annotated_frame, res = pipeline.process_display_frame(frame.copy())

                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Extract threat parameters from unified state
                    threat_level = res["threat_level"]
                    person_count = res["person_count"]
                    summary_reason = res["summary_reason"]
                    is_sos = res.get("sos_detected", False)
                    current_time = time.time()

                    # Dynamic credentials check
                    curr_token, curr_chat_id = get_telegram_credentials()

                    # 1. State Transition Logging
                    if threat_level != last_logged_state:
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        safe_log(f"[DEBUG THREAT TRIGGER] Threat state transition: {last_logged_state} -> {threat_level} at {timestamp_str}")
                        safe_log(f"[DEBUG THREAT DETAILS] Person Count: {person_count} | Trigger Reason: {summary_reason}")

                    # State transitions & persistence logging
                    if threat_level != last_logged_state or (current_time - last_logged_time > 10):
                        dsp_id = res.get("dispatch_payload", {}).get("dispatch_id") if res.get("dispatch_payload") else None
                        database.log_incident(threat_level, person_count, summary_reason, dsp_id)
                        last_logged_state = threat_level
                        last_logged_time = current_time
                        log_placeholder.markdown(render_db_activity_logs(), unsafe_allow_html=True)
                        render_hourly_chart()

                    # ----------------------------------------------------
                    # 2. GUARANTEED INSTANT SOS TELEGRAM DISPATCH
                    # ----------------------------------------------------
                    if is_sos:
                        if not last_sos_active:
                            safe_log(f"\n🚨 [CRITICAL SOS DETECTED] Fresh SOS Gesture Triggered! Dispatching instant Telegram Alert...")
                            dsp_id = res.get("dispatch_payload", {}).get("dispatch_id")
                            dispatch_telegram_alert_async(
                                threat_level="ALERT",
                                person_count=person_count,
                                trigger_reason=summary_reason,
                                dispatch_id=dsp_id,
                                bot_token=curr_token,
                                chat_id=curr_chat_id,
                                is_sos=True
                            )
                            last_sos_active = True
                            last_sos_telegram_time = current_time
                            last_telegram_state = "ALERT"
                        else:
                            if current_time - last_sos_telegram_time > 15.0:
                                safe_log(f"[DEBUG SOS REPEAT] SOS gesture sustained over 15s. Sending repeat dispatch...")
                                dsp_id = res.get("dispatch_payload", {}).get("dispatch_id")
                                dispatch_telegram_alert_async(
                                    threat_level="ALERT",
                                    person_count=person_count,
                                    trigger_reason=summary_reason,
                                    dispatch_id=dsp_id,
                                    bot_token=curr_token,
                                    chat_id=curr_chat_id,
                                    is_sos=True
                                )
                                last_sos_telegram_time = current_time
                    else:
                        last_sos_active = False

                        # Handle non-SOS alerts (e.g. Lone woman enclosed by men / proximity / crowding)
                        if threat_level == "ALERT":
                            elapsed_gen = current_time - last_telegram_time
                            if last_telegram_state != "ALERT" or (elapsed_gen > 60.0):
                                safe_log(f"[DEBUG STANDARD ALERT] Standard Alert triggered ({summary_reason}). Dispatching Telegram alert...")
                                dsp_id = res.get("dispatch_payload", {}).get("dispatch_id")
                                dispatch_telegram_alert_async(
                                    threat_level=threat_level,
                                    person_count=person_count,
                                    trigger_reason=summary_reason,
                                    dispatch_id=dsp_id,
                                    bot_token=curr_token,
                                    chat_id=curr_chat_id,
                                    is_sos=False
                                )
                                last_telegram_time = current_time
                                last_telegram_state = "ALERT"
                        else:
                            last_telegram_state = threat_level

                    # Update Threat Indicator Panel
                    if threat_level == "ALERT":
                        status_title = "STATUS: SOS EMERGENCY" if is_sos else "STATUS: ALERT"
                        threat_placeholder.markdown(
                            f"""
                            <div class="bento-card">
                                <div class="bento-card-title">🚨 Threat Level Indicator</div>
                                <div class="threat-status-box status-alert">
                                    <div style="font-size: 1.15rem; font-weight: 700;">{status_title}</div>
                                    <div style="font-size: 0.76rem; margin-top: 4px;">{summary_reason}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        disp = res.get("dispatch_payload")
                        if disp:
                            tg_status_note = "📱 Telegram Alert Dispatched (Priority 1)" if is_sos else ("📱 Telegram Alert Dispatched" if (curr_token and curr_chat_id) else "⚠️ Telegram Inactive")
                            dispatch_card_placeholder.markdown(
                                f"""
                                <div class="bento-card" style="border-left: 4px solid #EF4444; background: #FEF2F2;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="font-weight: 700; color: #991B1B; font-size: 0.9rem;">
                                            🚨 AUTOMATED EMERGENCY DISPATCH ACTIVE
                                        </div>
                                        <span style="font-size: 0.72rem; font-family: monospace; background: #FEE2E2; padding: 2px 8px; border-radius: 4px; color: #991B1B;">
                                            {disp['dispatch_id']}
                                        </span>
                                    </div>
                                    <div style="font-size: 0.8rem; color: #7F1D1D; margin-top: 6px;">
                                        • <strong>Recipient:</strong> {emergency_recipient}<br>
                                        • <strong>Location:</strong> {disp['location']}<br>
                                        • <strong>Trigger:</strong> {disp['primary_trigger']}<br>
                                        • <strong>Channel:</strong> {tg_status_note}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        if current_time - last_alert_beep > 2.0:
                            trigger_audio_alarm()
                            last_alert_beep = current_time

                    elif threat_level == "CAUTION":
                        dispatch_card_placeholder.empty()
                        threat_placeholder.markdown(
                            f"""
                            <div class="bento-card">
                                <div class="bento-card-title">🚨 Threat Level Indicator</div>
                                <div class="threat-status-box status-caution">
                                    <div style="font-size: 1.15rem; font-weight: 700;">STATUS: CAUTION</div>
                                    <div style="font-size: 0.76rem; margin-top: 4px;">{summary_reason}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        dispatch_card_placeholder.empty()
                        threat_placeholder.markdown(
                            f"""
                            <div class="bento-card">
                                <div class="bento-card-title">🚨 Threat Level Indicator</div>
                                <div class="threat-status-box status-safe">
                                    <div style="font-size: 1.15rem; font-weight: 700;">STATUS: SAFE</div>
                                    <div style="font-size: 0.76rem; margin-top: 4px;">{summary_reason}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Update Active Count Panel with Gender Breakdown
                    count_color = "#DC2626" if threat_level == "ALERT" else ("#D97706" if threat_level == "CAUTION" else "#111827")
                    g_breakdown = res.get("gender_breakdown", {})
                    f_num = g_breakdown.get("female", 0)
                    m_num = g_breakdown.get("male", 0)
                    gender_parts = []
                    if f_num > 0:
                        gender_parts.append(f"{f_num} Female")
                    if m_num > 0:
                        gender_parts.append(f"{m_num} Male")
                    gender_subtitle = f" | {', '.join(gender_parts)}" if gender_parts else ""
                    sos_status_text = "ACTIVE 🚨" if is_sos else (f"Tracking ({int(res.get('sos_progress', 0.0)*100)}%)" if res.get("sos_progress", 0.0) > 0 else "None")

                    count_placeholder.markdown(
                        f"""
                        <div class="bento-card">
                            <div class="bento-card-title">👥 Active People Count</div>
                            <div class="metric-value-large" style="color: {count_color};">{person_count}</div>
                            <div class="metric-subtitle">{res.get('time_factor_text', 'Standard')}{gender_subtitle} | SOS: {sos_status_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            finally:
                cap.release()
                pipeline.stop()
                cv2.destroyAllWindows()

    # ----------------------------------------------------
    # MODE B: PRE-RECORDED VIDEO UPLOAD (Demo Reliability)
    # ----------------------------------------------------
    elif video_source_mode == "Upload Video File (.mp4)":
        uploaded_file = st.file_uploader("Upload Demo Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            try:
                frame_count = 0
                while cap.isOpened() and run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    frame_count += 1
                    if frame_count % 2 == 0:
                        pipeline.submit_frame(frame)

                    annotated_frame, res = pipeline.process_display_frame(frame.copy())
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    threat_level = res["threat_level"]
                    person_count = res["person_count"]
                    summary_reason = res["summary_reason"]
                    is_sos = res.get("sos_detected", False)
                    current_time = time.time()
                    curr_token, curr_chat_id = get_telegram_credentials()

                    if is_sos:
                        if not last_sos_active:
                            dsp_id = res.get("dispatch_payload", {}).get("dispatch_id")
                            dispatch_telegram_alert_async(
                                threat_level="ALERT",
                                person_count=person_count,
                                trigger_reason=summary_reason,
                                dispatch_id=dsp_id,
                                bot_token=curr_token,
                                chat_id=curr_chat_id,
                                is_sos=True
                            )
                            last_sos_active = True
                            last_sos_telegram_time = current_time
                    else:
                        last_sos_active = False

                    count_placeholder.markdown(
                        f"""
                        <div class="bento-card">
                            <div class="bento-card-title">👥 Active People Count</div>
                            <div class="metric-value-large">{person_count}</div>
                            <div class="metric-subtitle">Demo Playback Mode</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    time.sleep(0.03)
            finally:
                cap.release()
                pipeline.stop()
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass
        else:
            video_placeholder.info("Please upload a video file above to start the demo playback.")
            pipeline.stop()

    # ----------------------------------------------------
    # MODE C: WEBRTC BROWSER STREAMING (Deployable)
    # ----------------------------------------------------
    elif video_source_mode == "Browser WebRTC (Cloud Ready)":
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
            import av

            class WebRTCAuraProcessor(VideoProcessorBase):
                def __init__(self):
                    self.pipeline = pipeline

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    self.pipeline.submit_frame(img)
                    annotated, _ = self.pipeline.process_display_frame(img)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_streamer(
                key="aura-webrtc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=WebRTCAuraProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        except Exception as e:
            st.error(f"WebRTC initialization: {e}")
            pipeline.stop()

else:
    # Idle Dashboard State
    video_placeholder.markdown(
        """
        <div class="bento-card" style="text-align: center; padding: 60px 20px; color: #6B7280;">
            <div style="font-size: 3.2rem; margin-bottom: 12px;">🛡️</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1F2937; margin-bottom: 6px;">
                Aura Vision Standby
            </div>
            <div style="font-size: 0.85rem; max-width: 420px; margin: 0 auto;">
                Flip the <strong>Camera Active</strong> toggle in the sidebar to activate the multi-threaded surveillance engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    threat_placeholder.markdown(
        """
        <div class="bento-card">
            <div class="bento-card-title">🚨 Threat Level Indicator</div>
            <div class="threat-status-box status-safe">
                <div style="font-size: 1.15rem; font-weight: 700;">STATUS: READY</div>
                <div style="font-size: 0.76rem; margin-top: 4px;">System standby — zero perimeter risk</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    count_placeholder.markdown(
        """
        <div class="bento-card">
            <div class="bento-card-title">👥 Active People Count</div>
            <div class="metric-value-large" style="color: #111827;">0</div>
            <div class="metric-subtitle">Standard Protocol | Standby</div>
        </div>
        """,
        unsafe_allow_html=True
    )
