import cv2
import threading
import requests
from flask import Flask, Response, jsonify

from motion_detection.MotionDetection_module import MotionDetection_module
from DNN_detection.DNN_module import DNN_module


# =========================
# Discord Config
# =========================
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1498469694095622214/TUCmlYmNiM47QJEPBgBoUQG0Jb0kYyNKohcKIq2fuZAaIIKRAPn3O-mkK9sQjcd5EsE8"
STREAM_URL = "http://100.89.44.99:5000/stream"

def send_detect_notification():
    try:
        msg = (
            "🐶 Dog detected!\n"
            f"View Stream:\n{STREAM_URL}"
        )
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": msg}
        )
    except Exception as e:
        print("Discord Error:", e)

def send_non_detect_notification():
    try:
        msg = ("Dog left.")
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": msg}
        )
    except Exception as e:
        print("Discord Error:", e)


# =========================
# Global Flags
# =========================
detect_flag = False
record_flag = False
latest_frame = None


# =========================
# Flask Server
# =========================
app = Flask(__name__)


@app.route("/record/start", methods=["POST"])
def start_record():
    global record_flag
    record_flag = True
    return jsonify({"record": True})


@app.route("/record/stop", methods=["POST"])
def stop_record():
    global record_flag
    record_flag = False
    return jsonify({"record": False})


@app.route("/status")
def status():
    return jsonify({
        "detect": detect_flag,
        "record": record_flag
    })


@app.route("/stream")
def stream():
    def generate():
        global latest_frame
        while True:
            if latest_frame is None:
                continue

            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame + b'\r\n')

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server():
    app.run(host="0.0.0.0", port=5000, threaded=True)


# =========================
# Detection config
# =========================
dnn_trial = 10
md_config = {
    "md_h": 30,
    "md_v": 24,
    "bufnum": 15, 
    "update_period": 5,
    "pix_thresh": 25,
    "num_thresh": 20,
}
dnn_config = {
    "target": "Dog", 
    "conf_thresh": 90,
}

# =========================
# Camera Init
# =========================
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
det_fps = 3
det_drop = max(1, int(fps / det_fps))

md = MotionDetection_module(md_config)
dnn = DNN_module(dnn_config)

frame_count = 0
prev_detect_flag = False


# =========================
# Start API Server
# =========================
threading.Thread(target=run_server, daemon=True).start()


# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- Detection ---
    if frame_count % det_drop == 0:
        motion = md.detect(frame)

        if motion:
            detect_flag = dnn.detect(frame)
        else:
            detect_flag = False

    # --- Discord Notify ---
    if detect_flag and not prev_detect_flag:
        send_detect_notification()

    elif not detect_flag and prev_detect_flag:
        send_non_detect_notification()

    prev_detect_flag = detect_flag

    # --- Output Resolution ---
    if record_flag:
        output_frame = cv2.resize(frame, (1920, 1080))
    elif detect_flag:
        output_frame = cv2.resize(frame, (1280, 720))
    else:
        output_frame = cv2.resize(frame, (640, 360))

    latest_frame = output_frame.copy()

    cv2.imshow("camera", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
