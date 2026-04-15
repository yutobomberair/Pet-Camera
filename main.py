import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# =========================
# モデルロード
# =========================
interpreter = tflite.Interpreter(model_path="training/yolov8n-cls_saved_model/yolov8n-cls_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# クラス
# =========================
classes = ["Dog", "Person", "Cat", "Bird"]

# =========================
# カメラ
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # 前処理
    # =========================
    img = cv2.resize(frame, (320, 320))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)

    # =========================
    # 推論
    # =========================
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # =========================
    # 後処理
    # =========================
    pred = np.argmax(output)
    conf = output[pred]

    label = classes[pred]

    print(f"{label}: {conf:.2f}")

    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
