import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

from motion_detection.MotionDetection_module import MotionDetection_module
from DNN_detection.DNN_module import DNN_module 

# =========================
# カメラ
# =========================
cap = cv2.VideoCapture(0)

# =========================
# config
# =========================
STATE = {0: "md", 1: "dnn", 2: "fr"}
det_fps = 10
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
# process
# =========================
if not cap.isOpened():
    print("❌ カメラ開けてない")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps)
drop = fps / det_fps

current_state = STATE[0]
frm_count = 0
dnn_frm_count = 0
md = MotionDetection_module(md_config)
dnn = DNN_module(dnn_config)

while True:
    if current_state != "fr" and frm_count < drop:
        frm_count += 1
        continue

    elif current_state == "md":
        print("state is md.")
        frm_count = 0
        ret, frame = cap.read()
        if not ret:
            print("❌ フレーム取得失敗")
            break
        is_detect = md.detect
        if is_detect:
            current_state = "dnn"
        # else:
        #     current_state = "md"
        continue

    elif current_state == "dnn":
        print("state is dnn")
        ret, frame = cap.read()
        if not ret:
            print("❌ フレーム取得失敗")
            break

        is_detect = dnn.detect(frame)
        if is_detect:
            current_state = "fr"
        elif dnn_frm_count < dnn_trial:
            dnn_frm_count += 1
            # curren_state = "dnn"
        else:
            dnn_frm_count = 0
            current_state = "md"
        continue

    elif current_state == "fr":
        print("state is fr")
        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
