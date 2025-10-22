# face_core.py
import os
import math
import uuid
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import face_recognition

from liveness.silent_spoof import SilentSpoofDetector
from extract_photo import extract_photo_from_pdf

# ---------- helpers ----------
def pick_best_face(face_locs, W, H):
    if not face_locs:
        return None
    cx, cy = W / 2, H / 2
    def score(bb):
        t, r, b, l = bb
        area = (r - l) * (b - t)
        fx, fy = (l + r) / 2, (t + b) / 2
        center_pen = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        return (area, -center_pen)
    return max(face_locs, key=score)

def eye_aspect_ratio(eye):
    p = np.array(eye, dtype=np.float32)
    A = np.linalg.norm(p[1] - p[5]); B = np.linalg.norm(p[2] - p[4]); C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C + 1e-6)

def detect_blink(landmarks, ear_thresh=0.21):
    le = landmarks["left_eye"]; re = landmarks["right_eye"]
    ear = (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0
    return ear < ear_thresh, ear

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),           # nose tip
        (0.0, -330.0, -65.0),      # chin
        (-225.0, 170.0, -135.0),   # left eye left
        (225.0, 170.0, -135.0),    # right eye right
        (-150.0, -150.0, -125.0),  # left mouth
        (150.0, -150.0, -125.0),   # right mouth
    ],
    dtype=np.float64,
)

def _landmark_points_2d(lmk):
    def c(pts): return np.mean(pts, axis=0)
    nose = c(lmk["nose_tip"]); chin = c(lmk["chin"])
    le = lmk["left_eye"][0]; re = lmk["right_eye"][-1]
    lm = lmk["top_lip"][0]; rm = lmk["top_lip"][-1]
    return np.array([nose, chin, le, re, lm, rm], dtype=np.float64)

def estimate_head_pose_yaw(frame, lmk):
    try:
        img_pts = _landmark_points_2d(lmk)
        h, w = frame.shape[:2]
        focal = w; center = (w / 2, h / 2)
        K = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype="double")
        ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, img_pts, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        return math.degrees(math.atan2(R[2,0], sy))
    except Exception:
        return None

def extract_known_encoding(path, out_img_path="images/resume_image.jpg"):
    low = path.lower()
    if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        img_path = path
    else:
        if not os.path.exists(path):
            return None
        if not extract_photo_from_pdf(path, out_img_path):
            return None
        img_path = out_img_path

    img = face_recognition.load_image_file(img_path)
    locs = face_recognition.face_locations(img, model="cnn") or face_recognition.face_locations(img, model="hog")
    if not locs:
        return None
    if len(locs) > 1:
        locs = [max(locs, key=lambda b: (b[2]-b[0])*(b[1]-b[3]))]
    encs = face_recognition.face_encodings(img, known_face_locations=locs, num_jitters=8, model="large")
    return encs[0] if len(encs) == 1 else None


# ---------- core ----------
class FaceVerifier:
    """FaceVerifier with configurable save semantics ('once' or 'cooldown')."""

    def __init__(
        self,
        resume_path="resume.pdf",
        passport_path=None,
        model_dir="models/anti_spoof",
        sim_th=0.62,
        live_th=0.45,
        live_win=5,
        require_blinks=1,
        require_yaw=15.0,
        accept_window=3,
        accept_need=2,
        prefer_model="hog",
        liveness_input="rgb",
        save_mode="once",            # "once" or "cooldown"
        save_cooldown_sec=5.0,       # cooldown for "cooldown" mode
    ):
        os.makedirs("matched_faces", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("images", exist_ok=True)

        self.known = extract_known_encoding(resume_path, "images/resume_image.jpg")
        self.known_pass = None
        if passport_path:
            self.known_pass = extract_known_encoding(passport_path, "images/passport_image.jpg")

        self.spoof = SilentSpoofDetector(model_dir=model_dir, threshold=live_th)
        print(f"[INFO] Liveness: loaded {len(getattr(self.spoof, 'models', []))} model(s) from: {os.path.abspath(model_dir)}")

        self.sim_th = sim_th
        self.live_th = live_th
        self.live_win = live_win
        self.require_blinks = require_blinks
        self.require_yaw = require_yaw
        self.accept_window = accept_window
        self.accept_need = accept_need
        self.prefer_model = prefer_model
        self.liveness_input = liveness_input

        # Save behavior
        assert save_mode in ("once", "cooldown"), "save_mode must be 'once' or 'cooldown'"
        self.save_mode = save_mode
        self.save_cooldown_sec = float(save_cooldown_sec)

        # session store
        self.sessions = {}  # sid -> state

    def create_session(self):
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "live_scores": deque(maxlen=max(1, self.live_win)),
            "live_avg": 0.0,
            "blink_consec": 0,
            "blinks": 0,
            "yaw_ref": None,
            "yaw_delta": 0.0,
            "votes": deque(maxlen=max(1, self.accept_window)),
            "accepted": False,
            "accepted_once": False,
            "best_frame": None,
            "saved_path": None,
            # save state
            "saved_once": False,
            "last_save_ts": 0.0,
            # replay etc (left simple here)
            "prev_gray": None,
        }
        return sid

    def finalize(self, sid, save=True):
        st = self.sessions.get(sid)
        if not st:
            return {"error": "invalid_session"}
        path = st["saved_path"]
        if save and st["accepted"] and st["best_frame"] is not None and path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"matched_faces/match_{ts}.jpg"
            cv2.imwrite(path, st["best_frame"])
            st["saved_path"] = path
        return {
            "accepted": st["accepted"],
            "live_avg": st["live_avg"],
            "blinks": st["blinks"],
            "yaw_delta": st["yaw_delta"],
            "saved_path": st["saved_path"],
        }

    def reset_session_saving(self, sid):
        st = self.sessions.get(sid)
        if not st:
            return False
        st["saved_once"] = False
        st["last_save_ts"] = 0.0
        return True

    def process_frame(self, sid, frame_bgr):
        st = self.sessions.get(sid)
        if not st:
            return {"error": "invalid_session"}

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # detect face
        locs = face_recognition.face_locations(rgb, model=self.prefer_model)
        if not locs:
            return {"status": "noface"}

        t, r, b, l = pick_best_face(locs, W, H)
        if t is None:
            return {"status": "noface"}

        # liveness
        inp = frame_bgr if self.liveness_input == "bgr" else rgb
        _, live_score = self.spoof.predict_is_live(inp, (l, t, r, b))
        st["live_scores"].append(float(live_score))
        st["live_avg"] = float(sum(st["live_scores"]) / len(st["live_scores"]))
        warmup = st["live_avg"] < self.live_th

        if warmup:
            return {
                "status": "live_warmup",
                "bbox": [int(l), int(t), int(r), int(b)],
                "live_avg": st["live_avg"],
                "blinks": st["blinks"],
                "yaw_delta": st["yaw_delta"],
                "prompt": "Proving live…",
            }

        # landmarks / blink / yaw
        lmk_list = face_recognition.face_landmarks(rgb, [(t, r, b, l)])
        if lmk_list:
            lmk = lmk_list[0]
            blink_now, _ = detect_blink(lmk)
            if blink_now:
                st["blink_consec"] += 1
            else:
                if st["blink_consec"] >= 2:
                    st["blinks"] += 1
                st["blink_consec"] = 0

            yaw = estimate_head_pose_yaw(frame_bgr, lmk)
            if yaw is not None:
                if st["yaw_ref"] is None:
                    st["yaw_ref"] = yaw
                st["yaw_delta"] = max(st["yaw_delta"], abs(yaw - st["yaw_ref"]))

        # challenges
        challenges_ok = (st["blinks"] >= self.require_blinks) and (st["yaw_delta"] >= self.require_yaw)
        if not challenges_ok:
            return {
                "status": "challenge",
                "bbox": [int(l), int(t), int(r), int(b)],
                "live_avg": st["live_avg"],
                "blinks": st["blinks"],
                "yaw_delta": st["yaw_delta"],
                "prompt": "Do a BLINK and TURN your head",
            }

        # recognition (compare to resume and passport if available)
        encs = face_recognition.face_encodings(rgb, [(t, r, b, l)])
        if not encs:
            return {"status": "noface"}
        enc = encs[0]

        distance = None
        match = False
        if self.known is not None:
            d = float(face_recognition.face_distance([self.known], enc)[0])
            distance = d
            if d < self.sim_th:
                match = True
                st["votes"].append(1)
            else:
                st["votes"].append(0)

        if self.known_pass is not None:
            d2 = float(face_recognition.face_distance([self.known_pass], enc)[0])
            # If both present, we record last distance (for info) but votes already tracked
            distance = d2 if distance is None else min(distance, d2)
            if d2 < self.sim_th:
                st["votes"].append(1)
            else:
                st["votes"].append(0)

        yes = int(sum(st["votes"]))
        st["accepted"] = bool(yes >= min(self.accept_need, len(st["votes"])) and any(v == 1 for v in st["votes"]))
        if st["accepted"] and not st["accepted_once"]:
            st["accepted_once"] = True

        info = {
            "status": "ok",
            "bbox": [int(l), int(t), int(r), int(b)],
            "live_avg": st["live_avg"],
            "blinks": st["blinks"],
            "yaw_delta": st["yaw_delta"],
            "match": st["accepted"],
            "distance": distance,
            "votes": yes,
            "accepted": st["accepted"],
            "saved_path": st["saved_path"],
        }

        # Save logic: obey save_mode
        now = time.time()
        saved_path = None
        if st["accepted"]:
            if self.save_mode == "once":
                if not st["saved_once"]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = f"matched_faces/match_{ts}.jpg"
                    cv2.imwrite(saved_path, frame_bgr)
                    st["saved_path"] = saved_path
                    st["saved_once"] = True
                    st["last_save_ts"] = now
            else:  # cooldown mode
                if (now - st["last_save_ts"]) > self.save_cooldown_sec:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = f"matched_faces/match_{ts}.jpg"
                    cv2.imwrite(saved_path, frame_bgr)
                    st["saved_path"] = saved_path
                    st["last_save_ts"] = now
                    st["saved_once"] = True

        if saved_path:
            info["saved_path"] = saved_path

        return info
