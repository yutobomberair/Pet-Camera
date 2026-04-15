import cv2
import numpy as np
import tensorflow as tf

# =========================
# モデルロード
# =========================
interpreter = tf.lite.Interpreter(
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

if not cap.isOpened():
    print("❌ カメラ開けてない")
    exit()

while True:
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
    if pred >= len(classes):
        label = f"Unknown({pred})"
    else:
        label = classes[pred]

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

    print(text)

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
