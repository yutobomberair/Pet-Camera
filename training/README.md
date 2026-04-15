# Dataset Preparation Guide (Classification Version)

このドキュメントでは、ペットカメラ用の**画像分類モデル**を学習するための  
**データセット作成手順**を説明します。

Open Images データセットから以下のクラスを取得し、  
**分類用（train / val / クラスフォルダ構造）データセット**を生成します。

* Dog  
* Person  
* Cat  
* Bird  

最終的に YOLO の分類モデル（`yolov8n-cls`）で  
そのまま学習できる構造を作成します。

---

# 1. 環境準備

```bash
python -m venv venv_train
source venv_train/bin/activate
```

```bash
pip install -r requirements4training.txt
```

---

# 2. データセット生成スクリプト

`scripts/prepare_dataset.py` を作成します。

※このスクリプトは  
**検出データ → 分類データへ自動変換**します

---

# 3. データセット作成

```bash
cd scripts
python prepare_dataset.py
```

---

# 4. 出力されるデータ構造

```
dataset_cls/
├── train/
│   ├── Dog/
│   ├── Person/
│   ├── Cat/
│   ├── Bird/
│
├── val/
│   ├── Dog/
│   ├── Person/
│   ├── Cat/
│   ├── Bird/
```

YOLO分類はこの構造をそのまま読み込みます。

---

# 5. データ数確認（任意）

```bash
ls dataset_cls/train/Dog | wc -l
ls dataset_cls/val/Dog | wc -l
```

---

# 6. 学習での利用例（分類）

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="../dataset_cls",
    epochs=20,
    imgsz=224,
    batch=32
)
```

---

# 7. 推論

```python
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

results = model("test.jpg")

print(results[0].probs)
```

---

# 8. モデル変換（予定）

```
PyTorch (.pt)
  ↓
ONNX
  ↓
TensorFlow
  ↓
TFLite
```

分類モデルにすることで：

- 出力が単純（Softmax）
- 後処理不要
- ONNX変換が安定
- ラズパイで軽量に動作

---

# Notes

## 現在の仕様

- 1画像に複数物体がある場合 → 最初の1つのみ使用
- bboxは未使用（画像全体で分類）

---

## 今後の改善

- bboxクロップ導入
- クラスバランス調整
- データ拡張（augment）

---

# Model Training Guide (Classification)

## 1. 学習スクリプト

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="/home/yuto/work/Pet-Camera/dataset_cls",
    epochs=20,
    imgsz=224,
    batch=32
)
```

---

## 2. 学習実行

```bash
cd training
python train.py
```

---

## 3. 出力

```
runs/classify/train/
```

```
runs/classify/train/weights/best.pt
```

---

## 4. 推論テスト

```python
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")

results = model("test.jpg", show=True)
```

---

## 5. 学習時間

| 環境 | 時間 |
|------|------|
| CPU | 数時間 |
| GPU | 約10〜20分 |

---

# 設計変更まとめ

| Before | After |
|------|------|
| 物体検出 | 画像分類 |
| YOLO detect | YOLO classify |
| bbox必要 | 不要 |
| 後処理複雑 | シンプル |
| 変換不安定 | 安定 |
