# liveness/silent_spoof.py
import os, re
import cv2
import numpy as np

# If you want to enable torch-based models, import torch and load .pth files as before.
try:
    import torch, torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# minimal heuristic-based fallback
class SilentSpoofDetector:
    def __init__(self, model_dir="models/anti_spoof", threshold=0.80, device=None):
        self.model_dir = model_dir
        self.threshold = float(threshold)
        self.models = []
        self.device = device
        # Optionally load torch models if available
        if _TORCH_OK and os.path.isdir(model_dir):
            # If you want to auto-load .pth like in your full file, keep that logic.
            pass

    def _heuristic_score(self, bgr_face):
        if bgr_face is None or bgr_face.size == 0:
            return 0.0
        gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        sharp_var = float(np.var(lap))
        sharp_score = np.clip((sharp_var - 80.0) / 220.0, 0.0, 1.0)
        hsv = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv[..., 1]))
        sat_score = np.clip((sat_std - 10.0) / 40.0, 0.0, 1.0)
        # motion score is handled at caller if needed; here assume neutral 0.5
        motion_score = 0.5
        raw = 0.45 * sharp_score + 0.30 * sat_score + 0.25 * motion_score
        return float(np.clip(raw, 0.0, 1.0))

    def predict_is_live(self, frame_bgr, bbox_xyxy, margin=0.25):
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = frame_bgr.shape[:2]
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        x1 = max(0, x1 - int(bw * margin)); y1 = max(0, y1 - int(bh * margin))
        x2 = min(w, x2 + int(bw * margin)); y2 = min(h, y2 + int(bh * margin))
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return False, 0.0

        # If torch-based models were loaded, you could infer here. For now use heuristic:
        score = self._heuristic_score(face)
        return (score >= self.threshold), float(score)
