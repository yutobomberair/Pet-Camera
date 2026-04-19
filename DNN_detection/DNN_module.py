import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

base_dir = os.path.dirname(__file__)

class DNN_module:
    def __init__(self, config):
        self.interpreter = tflite.Interpreter(
            model_path=base_dir + "/model/classification.tflite"  # ← 必ず自分のbestモデルから変換したやつ
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("INPUT SHAPE:", self.input_details[0]['shape'])
        print("OUTPUT SHAPE:", self.output_details[0]['shape'])

        self.classes = ["Dog", "Person", "Cat", "Bird"]
        self.target = "Dog"
        self.conf_thresh = config["conf_thresh"]

    def preprocess(self, img):
        bin_img = cv2.resize(img, (320, 320))
        quant_img = bin_img.astype(np.float32) / 255.0
        return np.expand_dims(quant_img, axis=0)  # (1, 320, 320, 3)

    def detect(self, img):
        proc_img = self.preprocess(img)
        self.interpreter.set_tensor(self.input_details[0]['index'], proc_img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return self.postprocess(output)

    def postprocess(self, output):
        pred = int(np.argmax(output))
        conf = float(output[pred])
        # 安全対策（1000クラス問題防止）
        if pred < len(self.classes):
            label = self.classes[pred]
        else:
            label = "Unknown"
        is_detect = 0
        if label == self.target and conf > self.conf_thresh:
            is_detect = 1
        return is_detect