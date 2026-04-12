# Dataset Preparation Guide

このドキュメントでは、ペットカメラ用の物体検出モデルを学習するための
**データセット作成手順**を説明します。

Open Images データセットから以下のクラスを取得し、YOLO形式の
`train / val` データセットを生成します。

* Dog
* Person
* Cat
* Bird

最終的に YOLO 系モデルでそのまま学習できる構造を作成します。

---

# 1. 環境準備

Python 仮想環境を作成し、必要なライブラリをインストールします。

```bash
python -m venv venv_train
source venv_train/bin/activate
```

必要ライブラリ：

```bash
pip install fiftyone scikit-learn
```

---

# 2. データセット生成スクリプト

`scripts/prepare_dataset.py` を作成します。

```python
import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from sklearn.model_selection import train_test_split

EXPORT_DIR = "../dataset"

CLASS_SAMPLES = {
    "Dog": 8000,
    "Person": 6000,
    "Cat": 2000,
    "Bird": 1500
}

# =========================
# dataset download
# =========================

datasets = []

for cls, n in CLASS_SAMPLES.items():

    print(f"Downloading {cls}: {n}")

    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[cls],
        max_samples=n,
        shuffle=True
    )

    datasets.append(ds)


dataset = fo.Dataset("pet_camera_dataset")

for ds in datasets:
    dataset.merge_samples(ds)

print("Total samples:", len(dataset))


# =========================
# export
# =========================

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=list(CLASS_SAMPLES.keys()),
)

print("Export complete")


# =========================
# split train / val
# =========================

SRC_IMG = os.path.join(EXPORT_DIR, "images", "val")
SRC_LBL = os.path.join(EXPORT_DIR, "labels", "val")

TRAIN_IMG = os.path.join(EXPORT_DIR, "images", "train")
VAL_IMG = os.path.join(EXPORT_DIR, "images", "val")

TRAIN_LBL = os.path.join(EXPORT_DIR, "labels", "train")
VAL_LBL = os.path.join(EXPORT_DIR, "labels", "val")

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)

images = [
    f for f in os.listdir(SRC_IMG)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Images found:", len(images))

train_imgs, val_imgs = train_test_split(
    images,
    test_size=0.1,
    random_state=42
)


def move(files, img_dst, lbl_dst):

    for f in files:

        name = os.path.splitext(f)[0]

        shutil.move(
            os.path.join(SRC_IMG, f),
            os.path.join(img_dst, f)
        )

        shutil.move(
            os.path.join(SRC_LBL, name + ".txt"),
            os.path.join(lbl_dst, name + ".txt")
        )


move(train_imgs, TRAIN_IMG, TRAIN_LBL)

print("Train images:", len(train_imgs))
print("Val images:", len(val_imgs))


# =========================
# dataset.yaml
# =========================

yaml = f"""path: {EXPORT_DIR}

train: images/train
val: images/val

names:
"""

for i, name in enumerate(CLASS_SAMPLES.keys()):
    yaml += f"  {i}: {name}\n"

with open(os.path.join(EXPORT_DIR, "dataset.yaml"), "w") as f:
    f.write(yaml)

print("dataset.yaml created")
```

---

# 3. データセット作成

以下を実行します。

```bash
cd scripts
python prepare_dataset.py
```

---

# 4. 出力されるデータ構造

処理が完了すると、以下の構造のデータセットが生成されます。

```
dataset/
 ├ images/
 │   ├ train/
 │   └ val/
 │
 ├ labels/
 │   ├ train/
 │   └ val/
 │
 └ dataset.yaml
```

---

# 5. データ数確認（任意）

```bash
ls dataset/images/train | wc -l
ls dataset/images/val | wc -l
```

ラベル数も一致していることを確認します。

```bash
ls dataset/labels/train | wc -l
```

---

# 6. 学習での利用例

YOLO 系モデルでは `dataset.yaml` を指定することで
そのまま学習に利用できます。

例：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../dataset/dataset.yaml",
    epochs=50,
    imgsz=640
)
```

---

## Notes

* Open Images データセットは画像サイズやラベル品質にばらつきがあります
* 必要に応じてクラス数やサンプル数を調整してください
* ペットカメラ用途では `Dog` と `Person` のみでも十分な場合があります


# Model Training Guide

このドキュメントでは、作成したデータセットを用いて\
物体検出モデルを学習する手順を説明します。

本プロジェクトでは YOLO ベースの軽量モデルを使用します。

------------------------------------------------------------------------

## 1. 環境準備

Python 仮想環境を作成します。

``` bash
python -m venv venv_train
source venv_train/bin/activate
```

必要なライブラリをインストールします。

``` bash
pip install ultralytics
```

------------------------------------------------------------------------

## 2. 学習スクリプト作成

`training/train.py` を作成します。

``` python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="/home/yuto/work/Pet-Camera/dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

------------------------------------------------------------------------

## 3. dataset.yaml の確認

``` yaml
path: /home/yuto/work/Pet-Camera/dataset

train: images/train
val: images/val

names:
  0: Dog
  1: Person
  2: Cat
  3: Bird
```

------------------------------------------------------------------------

## 4. 学習実行

``` bash
cd training
python train.py
```

------------------------------------------------------------------------

## 5. 学習結果

    runs/detect/train/

    runs/detect/train/weights/best.pt
    runs/detect/train/weights/last.pt

------------------------------------------------------------------------

## 6. 推論テスト

``` python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("test.jpg", show=True)
```

------------------------------------------------------------------------

## 7. 注意事項

    device=cpu

または

    CUDA initialization: The NVIDIA driver is too old

------------------------------------------------------------------------

## 8. 学習時間

  環境   学習時間
  ------ ------------
  CPU    数時間
  GPU    約20〜30分

