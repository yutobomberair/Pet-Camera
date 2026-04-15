import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

EXPORT_DIR = "../dataset"
CLS_DIR = "../dataset_cls"

CLASS_SAMPLES = {
    "Dog": 8000,
    "Person": 6000,
    "Cat": 2000,
    "Bird": 1500
}

# =========================
# ① dataset download
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
# ② YOLO形式でexport（中間）
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
# ③ YOLO検出 → 分類用に変換
# =========================

if os.path.exists(CLS_DIR):
    shutil.rmtree(CLS_DIR)

os.makedirs(CLS_DIR, exist_ok=True)

label_dir = os.path.join(EXPORT_DIR, "labels", "val")
img_dir = os.path.join(EXPORT_DIR, "images", "val")

class_map = defaultdict(list)

labels = os.listdir(label_dir)

for label_file in labels:

    label_path = os.path.join(label_dir, label_file)

    with open(label_path) as f:
        lines = f.readlines()

    if len(lines) == 0:
        continue

    # 1つ目の物体だけ使う（簡略）
    cls_id = int(lines[0].split()[0])
    cls_name = list(CLASS_SAMPLES.keys())[cls_id]

    img_name = label_file.replace(".txt", ".jpg")
    class_map[cls_name].append(img_name)


# =========================
# ④ train / val split（分類形式）
# =========================

for cls, imgs in class_map.items():

    train_imgs, val_imgs = train_test_split(
        imgs,
        test_size=0.1,
        random_state=42
    )

    # ディレクトリ作成
    train_dir = os.path.join(CLS_DIR, "train", cls)
    val_dir = os.path.join(CLS_DIR, "val", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # コピー関数
    def copy_images(file_list, dst_dir):
        for img in file_list:
            src = os.path.join(img_dir, img)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst_dir, img))

    copy_images(train_imgs, train_dir)
    copy_images(val_imgs, val_dir)

    print(f"{cls}: train={len(train_imgs)}, val={len(val_imgs)}")


print("Classification dataset created at:", CLS_DIR)

