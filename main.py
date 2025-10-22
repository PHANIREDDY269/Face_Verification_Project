# main.py
#!/usr/bin/env python3
import argparse
import time
import cv2

from face_core import FaceVerifier

# Optional Windows toast (pip install win10toast)
try:
    from win10toast import ToastNotifier
    _toaster = ToastNotifier()
except Exception:
    _toaster = None

# Optional beep fallback
try:
    import winsound
    def _beep(): winsound.Beep(1000, 300)
except Exception:
    def _beep(): pass

def draw_box(frame, bbox, color=(0, 255, 0), label: str = ""):
    l, t, r, b = map(int, bbox)
    cv2.rectangle(frame, (l, t), (r, b), color, 2)
    if label:
        cv2.putText(frame, label, (l, max(22, t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    ap = argparse.ArgumentParser(description="Live face verification with liveness + anti-replay challenges")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--resume", type=str, required=True, help="Path to resume PDF or photo")
    ap.add_argument("--passport", type=str, default=None, help="Path to passport photo (optional)")
    ap.add_argument("--model-dir", type=str, default="models/anti_spoof")
    ap.add_argument("--sim-th", type=float, default=0.62)
    ap.add_argument("--live-th", type=float, default=0.45)
    ap.add_argument("--live-win", type=int, default=5)
    ap.add_argument("--require-blinks", type=int, default=1)
    ap.add_argument("--require-yaw", type=float, default=15.0)
    ap.add_argument("--liveness-input", type=str, default="rgb", choices=["bgr", "rgb"])
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--save-mode", choices=["once", "cooldown"], default="once",
                    help="Saving behavior on acceptance: 'once' or 'cooldown'.")
    ap.add_argument("--save-cooldown", type=float, default=5.0,
                    help="Cooldown seconds between saves (only for cooldown mode).")
    ap.add_argument("--no-toast", action="store_true", help="Disable Windows toast notifications")
    ap.add_argument("--no-beep", action="store_true", help="Disable beep on accept")
    args = ap.parse_args()

    ver = FaceVerifier(
        resume_path=args.resume,
        passport_path=args.passport,
        model_dir=args.model_dir,
        sim_th=args.sim_th,
        live_th=args.live_th,
        live_win=args.live_win,
        require_blinks=args.require_blinks,
        require_yaw=args.require_yaw,
        accept_window=3,
        accept_need=2,
        prefer_model="hog",
        liveness_input=args.liveness_input,
        save_mode=args.save_mode,
        save_cooldown_sec=args.save_cooldown,
    )

    sid = ver.create_session()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.cam}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("q=quit   [ / ] adjust live_th   r=reset saved flag")
    accepted_notified = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        res = ver.process_frame(sid, frame)
        view = frame.copy()

        if res.get("error"):
            cv2.putText(view, f"Error: {res['error']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status = res.get("status")
        if status == "noface":
            cv2.putText(view, "No face detected — move closer / adjust lighting", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif status == "live_warmup":
            draw_box(view, res["bbox"], (0, 165, 255),
                     f"Proving live... avg={res['live_avg']:.2f}")

        elif status == "challenge":
            label = (f"{res.get('prompt','')} | live {res.get('live_avg',0):.2f} "
                     f"| blinks:{res.get('blinks',0)} yaw={res.get('yaw_delta',0):.1f}")
            draw_box(view, res["bbox"], (255, 140, 0), label)

        else:
            bbox = res.get("bbox", None)
            if bbox:
                color = (0, 255, 0) if res.get("accepted") else (0, 0, 255)
                label = "Match" if res.get("accepted") else "No Match"
                if res.get("distance") is not None:
                    label += f" ({res['distance']:.2f})"
                label += f" | live {res.get('live_avg',0):.2f} | votes {res.get('votes',0)}"
                draw_box(view, bbox, color, label)

            if res.get("accepted"):
                cv2.putText(view, "ACCEPTED ✓", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                if not accepted_notified:
                    if not args.no_beep:
                        try: _beep()
                        except Exception: pass
                    if _toaster and not args.no_toast:
                        msg = "Face verified"
                        if res.get("saved_path"):
                            msg += f" — saved: {res['saved_path']}"
                        try:
                            _toaster.show_toast("Face Verification", msg, duration=4, threaded=True)
                        except Exception: pass
                    accepted_notified = True

        fps = 1.0 / max(1e-6, (time.time() - t0))
        cv2.putText(view, f"FPS: {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, f"live_th: {ver.live_th:.2f}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Verify", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('['):
            ver.live_th = max(0.20, ver.live_th - 0.02)
            print(f"live_th -> {ver.live_th:.2f}")
        elif key == ord(']'):
            ver.live_th = min(0.95, ver.live_th + 0.02)
            print(f"live_th -> {ver.live_th:.2f}")
        elif key == ord('r'):
            # reset saved flag for current session (useful while testing)
            ver.reset_session_saving(sid)
            accepted_notified = False
            print("Saved flag reset — ready to save again.")

    cap.release()
    cv2.destroyAllWindows()
    print(ver.finalize(sid, save=True))


if __name__ == "__main__":
    main()
