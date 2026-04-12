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

yaml = f"""path: {EXPORT_DIR.split('/')[-1]}

train: images/train
val: images/val

names:
"""

for i, name in enumerate(CLASS_SAMPLES.keys()):
    yaml += f"  {i}: {name}\n"

with open(os.path.join(EXPORT_DIR, "dataset.yaml"), "w") as f:
    f.write(yaml)

print("dataset.yaml created")
