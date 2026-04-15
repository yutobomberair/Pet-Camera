from ultralytics import YOLO

# 分類モデル読み込み（軽いモデル）
model = YOLO("yolov8n-cls.pt")

# 学習
model.train(
    data="dataset_cls",   # ← ルートディレクトリ
    epochs=50,
    imgsz=224,
    batch=32,
    device="cuda"
)
