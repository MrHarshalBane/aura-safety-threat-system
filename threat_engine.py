import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class ThreatAnalysisEngine:
    """
    Multi-Signal Threat Analysis Engine for Aura Safety.
    Evaluates:
    - SOS Hand / Arm gestures from YOLO pose keypoints
    - Rapid motion / struggle
    - Night-time lone woman vulnerability vs surrounding individuals
    - Proximity / crowding alerts
    """
    def __init__(self, required_sos_frames: int = 2, max_missed_frames: int = 4):
        self.required_sos_frames = required_sos_frames
        self.max_missed_frames = max_missed_frames
        self.sos_counter = 0
        self.missed_counter = 0
        self.last_alert_time = 0

    def check_sos_gesture(self, keypoints) -> bool:
        """
        Detects raised hands / distress signal using YOLO pose keypoints.
        Keypoints format: COCO keypoints:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist
        """
        if keypoints is None:
            return False

        try:
            # Handle Ultralytics Keypoints object or ndarray
            if hasattr(keypoints, "xy"):
                kp_data = keypoints.xy.cpu().numpy()
            elif isinstance(keypoints, np.ndarray):
                kp_data = keypoints
            else:
                return False

            if len(kp_data) == 0:
                return False

            for person_kp in kp_data:
                if len(person_kp) < 11:
                    continue
                
                # Check wrists above shoulders or nose (hands raised in distress / surrender / SOS)
                nose = person_kp[0]
                left_shoulder = person_kp[5]
                right_shoulder = person_kp[6]
                left_wrist = person_kp[9]
                right_wrist = person_kp[10]

                # If wrist y is significantly smaller than shoulder y (y is inverted in screen space)
                left_raised = (left_wrist[1] > 0 and left_shoulder[1] > 0 and left_wrist[1] < left_shoulder[1])
                right_raised = (right_wrist[1] > 0 and right_shoulder[1] > 0 and right_wrist[1] < right_shoulder[1])

                if left_raised or right_raised:
                    return True
        except Exception:
            return False

        return False

    def evaluate_threat(
        self,
        boxes: np.ndarray,
        keypoints: Any = None,
        frame_shape: Tuple[int, int] = (480, 640),
        simulated_hour: Optional[int] = None,
        gender_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Multi-signal fusion evaluating threat level:
        Returns:
            threat_level: "SAFE" | "CAUTION" | "ALERT"
            person_count: int
            gender_breakdown: Dict
            summary_reason: str
            sos_detected: bool
            sos_progress: float
            dispatch_payload: Dict
        """
        person_count = len(boxes) if boxes is not None else 0
        h, w = frame_shape[:2]

        hour = simulated_hour if simulated_hour is not None else datetime.now().hour
        is_night = (hour >= 20 or hour < 6)
        time_factor_text = f"Night Vigilance Protocol ({hour:02d}:00)" if is_night else f"Day Protocol ({hour:02d}:00)"

        # Count genders
        gender_breakdown = {"female": 0, "male": 0, "unknown": 0}
        if gender_data:
            for item in gender_data:
                g = item.get("gender", "Unknown").lower()
                if g in gender_breakdown:
                    gender_breakdown[g] += 1
                else:
                    gender_breakdown["unknown"] += 1

        # Check SOS Gesture
        has_sos_gesture = self.check_sos_gesture(keypoints)
        if has_sos_gesture:
            self.sos_counter += 1
            self.missed_counter = 0
        else:
            if self.sos_counter > 0:
                self.missed_counter += 1
                if self.missed_counter > self.max_missed_frames:
                    self.sos_counter = 0
                    self.missed_counter = 0

        sos_detected = self.sos_counter >= self.required_sos_frames
        sos_progress = min(1.0, self.sos_counter / float(max(1, self.required_sos_frames)))

        threat_level = "SAFE"
        summary_reason = "Perimeter Secure — Normal Activity"

        # Threat Rules
        if sos_detected:
            threat_level = "ALERT"
            summary_reason = "DISTRESS GESTURE (Hands Raised / SOS Confirmed)"
        elif is_night and gender_breakdown.get("female", 0) >= 1 and gender_breakdown.get("male", 0) >= 2:
            threat_level = "ALERT"
            summary_reason = f"Lone Woman surrounded by {gender_breakdown.get('male')} males at night"
        elif is_night and gender_breakdown.get("female", 0) == 1 and person_count == 1:
            threat_level = "CAUTION"
            summary_reason = "Vulnerable Lone Female Detected in Perimeter (Night Watch)"
        elif person_count >= 5:
            threat_level = "CAUTION"
            summary_reason = f"High Density Crowding Detected ({person_count} persons)"
        elif person_count > 0:
            summary_reason = f"Monitoring {person_count} active person(s)"

        dispatch_payload = None
        if threat_level == "ALERT":
            dispatch_payload = {
                "dispatch_id": f"AURA-{int(time.time()) % 100000:05d}",
                "location": "Sector 4 — Main Surveillance Zone",
                "primary_trigger": summary_reason,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        return {
            "threat_level": threat_level,
            "person_count": person_count,
            "gender_breakdown": gender_breakdown,
            "summary_reason": summary_reason,
            "sos_detected": sos_detected,
            "sos_progress": sos_progress,
            "prox_alert": (person_count > 3),
            "motion_alert": False,
            "is_night": is_night,
            "time_factor_text": time_factor_text,
            "dispatch_payload": dispatch_payload
        }
