import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

from motion_dtection.MotionDetection_module import MotionDetection_module 

# =========================
# モデルロード
# =========================
interpreter = tflite.Interpreter(
    model_path="training/yolov8n-cls_float32.tflite"  # ← 必ず自分のbestモデルから変換したやつ
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT SHAPE:", input_details[0]['shape'])
print("OUTPUT SHAPE:", output_details[0]['shape'])

# =========================
# クラス
# =========================
classes = ["Dog", "Person", "Cat", "Bird"]

# =========================
# カメラ
# =========================
cap = cv2.VideoCapture(0)

# =========================
# config
# =========================
STATE = {0: "md", 1: "dnn", 2: "fr"}
det_fps = 10
conf_thresh = 90
md_configs = {
    "md_h": 30,
    "md_v": 24,
    "bufnum": 25, 
    "update_period": 5,
    "pix_thresh": 25,
    "num_thresh": 20,
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
md = MotionDetection_module()
while True:
    if current_state != "fr" and frm_count < drop:
        frm_count += 1
        continue

    elif current_state == "md":
        frm_count = 0
        ret, frame = cap.read()
        if not ret:
            print("❌ フレーム取得失敗")
            break
        md.update_buffer(frame)
        current_state = "dnn"
        print("a")
        continue

    elif current_state == "dnn":
        ret, frame = cap.read()
        if not ret:
            print("❌ フレーム取得失敗")
            break

        # =========================
        # 前処理
        # =========================
        img = cv2.resize(frame, (320, 320))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 320, 320, 3)

        # =========================
        # 推論
        # =========================
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # =========================
        # 後処理
        # =========================
        pred = int(np.argmax(output))
        conf = float(output[pred])

        # 安全対策（1000クラス問題防止）
        if pred < len(classes):
            label = classes[pred]
        else:
            label = "Unknown"

        if label == "Dog" and conf >= conf_thresh:
            current_state = "fr"

        # =========================
        # 描画
        # =========================
        text = f"{label}: {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # print(text)

        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    count = 0

cap.release()
cv2.destroyAllWindows()
