import os
import cv2
import numpy as np
import urllib.request
from typing import Tuple, List, Optional, Dict, Any

GENDER_PROTOTXT_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
GENDER_CAFFE_MODEL_URL = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"

GENDER_LIST = ["Male", "Female"]
# Mean subtraction values for Levi & Hassner Caffe Model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

class GenderClassifier:
    """
    Lightweight Caffe-based Gender Classification Engine using OpenCV DNN.
    Pretrained Levi & Hassner CNN model.
    """
    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir
        self.prototxt_path = os.path.join(model_dir, "deploy_gender.prototxt")
        self.caffemodel_path = os.path.join(model_dir, "gender_net.caffemodel")
        self.net: Optional[cv2.dnn.Net] = None
        self.is_loaded: bool = False
        
        self.load_model()

    def download_model_files(self) -> bool:
        """Attempts to automatically download the Caffe model files if missing."""
        try:
            if not os.path.exists(self.prototxt_path):
                print(f"[DEBUG GENDER] Downloading deploy_gender.prototxt to {self.prototxt_path}...")
                urllib.request.urlretrieve(GENDER_PROTOTXT_URL, self.prototxt_path)
                
            if not os.path.exists(self.caffemodel_path):
                print(f"[DEBUG GENDER] Downloading gender_net.caffemodel to {self.caffemodel_path}...")
                urllib.request.urlretrieve(GENDER_CAFFE_MODEL_URL, self.caffemodel_path)
                
            return os.path.exists(self.prototxt_path) and os.path.exists(self.caffemodel_path)
        except Exception as e:
            print(f"[DEBUG GENDER] Auto-download failed: {e}. Model files must be placed locally.")
            return False

    def load_model(self) -> bool:
        """Loads the Caffe model into OpenCV DNN."""
        if not (os.path.exists(self.prototxt_path) and os.path.exists(self.caffemodel_path)):
            self.download_model_files()

        if os.path.exists(self.prototxt_path) and os.path.exists(self.caffemodel_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.is_loaded = True
                print("[DEBUG GENDER] Caffe Gender Classifier model loaded successfully.")
                return True
            except Exception as e:
                print(f"[DEBUG GENDER ERROR] Failed to load Caffe model: {e}")
                self.is_loaded = False
                return False
        else:
            print("[DEBUG GENDER] Model files not found. Gender classification will operate in fallback mode.")
            self.is_loaded = False
            return False

    def classify_crop(self, crop: np.ndarray, min_confidence: float = 0.65) -> Tuple[str, float]:
        """
        Classifies the gender of an individual crop.
        Returns ('Male'|'Female'|'Unknown', confidence).
        """
        if not self.is_loaded or self.net is None or crop is None or crop.size == 0:
            return "Unknown", 0.0

        try:
            h, w = crop.shape[:2]
            if h < 20 or w < 20:
                return "Unknown", 0.0

            # Preprocess image crop for Levi-Hassner model (227x227, swapRB=False)
            blob = cv2.dnn.blobFromImage(
                crop,
                scalefactor=1.0,
                size=(227, 227),
                mean=MODEL_MEAN_VALUES,
                swapRB=False,
                crop=False
            )
            self.net.setInput(blob)
            gender_preds = self.net.forward()
            
            idx = int(gender_preds[0].argmax())
            conf = float(gender_preds[0][idx])
            label = GENDER_LIST[idx]

            if conf >= min_confidence:
                return label, conf
            else:
                return "Unknown", conf
        except Exception:
            return "Unknown", 0.0

    def classify_person_box(self, frame: np.ndarray, box: np.ndarray, min_confidence: float = 0.65) -> Tuple[str, float]:
        """
        Extracts upper-body / face region from a person bounding box and runs classification.
        """
        if not self.is_loaded or frame is None or len(box) < 4:
            return "Unknown", 0.0

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Clamp to frame boundary
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < 30 or box_w < 20:
            return "Unknown", 0.0

        # Extract head & upper torso region (upper 45% of person bounding box)
        crop_y2 = y1 + int(box_h * 0.45)
        person_crop = frame[y1:crop_y2, x1:x2]

        if person_crop.size == 0 or person_crop.shape[0] < 15 or person_crop.shape[1] < 15:
            person_crop = frame[y1:y2, x1:x2]

        return self.classify_crop(person_crop, min_confidence=min_confidence)

    def classify_all_persons(self, frame: np.ndarray, boxes: np.ndarray, min_confidence: float = 0.65) -> List[Dict[str, Any]]:
        """
        Classifies all detected person bounding boxes in a frame.
        Returns list of {'gender': label, 'confidence': conf, 'box': box}.
        """
        results = []
        if boxes is None or len(boxes) == 0:
            return results

        for box in boxes:
            gender, conf = self.classify_person_box(frame, box, min_confidence=min_confidence)
            results.append({
                "gender": gender,
                "confidence": conf,
                "box": box
            })
        return results
