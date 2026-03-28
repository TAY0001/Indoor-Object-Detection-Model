import os
import cv2
import random
import uuid
from collections import defaultdict
import albumentations as A

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/yolo_all2"
SPLIT = "train"  # augment TRAIN only
IMG_DIR = os.path.join(DATA_ROOT, SPLIT, "images")
LBL_DIR = os.path.join(DATA_ROOT, SPLIT, "labels")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TARGET_PER_CLASS = 1200
MAX_TRIES_MULTIPLIER = 6

random.seed(42)

# =========================
# AUGMENTATION PIPELINE
# YOLO bboxes: (class_id, x_center, y_center, w, h) normalized
# =========================

transform = A.Compose(
    [
        # ---- Geometry ----
        A.OneOf(
            [
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.90, 1.10),
                    rotate=(-12, 12),
                    shear=(-6, 6),
                    p=1.0,
                ),
                A.Perspective(scale=(0.02, 0.05), keep_size=True, p=1.0),
            ],
            p=0.85,
        ),

        # ---- Lighting / color ----
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=10, p=0.35),
        A.RandomGamma(gamma_limit=(85, 115), p=0.25),

        # ---- Noise / blur ----
        A.OneOf(
            [
                A.GaussNoise(p=1.0),                 
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ],
            p=0.25,
        ),

        # ---- Occlusion robustness ----
        A.CoarseDropout(p=0.20),                   

        # ---- Flip ----
        A.HorizontalFlip(p=0.35),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=16,
        min_visibility=0.25,
        clip=True,
    ),
)

# =========================
# HELPERS
# =========================
def list_images():
    files = []
    for f in os.listdir(IMG_DIR):
        if f.lower().endswith(IMAGE_EXTS):
            base = os.path.splitext(f)[0]
            lbl_path = os.path.join(LBL_DIR, base + ".txt")
            if os.path.exists(lbl_path):
                files.append(f)
    return files

def read_yolo_label(path):
    bboxes = []
    class_labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            bboxes.append([x, y, w, h])
            class_labels.append(cls)
    return bboxes, class_labels

def write_yolo_label(path, bboxes, class_labels):
    with open(path, "w") as f:
        for bb, cls in zip(bboxes, class_labels):
            x, y, w, h = bb
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def count_images_per_class(image_files):
    class_to_images = defaultdict(list)
    class_counts = defaultdict(int)

    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(LBL_DIR, base + ".txt")

        bboxes, class_labels = read_yolo_label(lbl_path)
        present = set(class_labels)

        for c in present:
            class_to_images[c].append(img_file)

        # count as "images containing class" (not instance count)
        for c in present:
            class_counts[c] += 1

    return class_counts, class_to_images

def unique_out_name(original_base):
    # keeps traceability, avoids collisions
    return f"{original_base}_aug_{uuid.uuid4().hex[:10]}"

# =========================
# MAIN
# =========================
image_files = list_images()
class_counts, class_to_images = count_images_per_class(image_files)

all_classes = sorted(class_counts.keys())
print("📊 Current train image counts per class (images containing class):")
for c in all_classes:
    print(f"  class {c}: {class_counts[c]}")

print(f"\n🎯 Target per class: {TARGET_PER_CLASS}")
print("🚀 Starting augmentation...\n")

aug_made_per_class = defaultdict(int)

for c in all_classes:
    current = class_counts[c]
    if current >= TARGET_PER_CLASS:
        continue

    need = TARGET_PER_CLASS - current
    pool = class_to_images[c]
    if not pool:
        print(f"⚠️ class {c}: no images found, skipping")
        continue

    tries_left = need * MAX_TRIES_MULTIPLIER
    made = 0

    while made < need and tries_left > 0:
        tries_left -= 1

        img_file = random.choice(pool)
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, base + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            continue

        bboxes, class_labels = read_yolo_label(lbl_path)
        if not bboxes:
            continue

        # Apply transform
        try:
            out = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            continue

        out_img = out["image"]
        out_bboxes = out["bboxes"]
        out_labels = out["class_labels"]

        # Must still contain the target class c
        if c not in set(out_labels):
            continue

        # Save
        new_base = unique_out_name(base)
        new_img_path = os.path.join(IMG_DIR, new_base + ".jpg")
        new_lbl_path = os.path.join(LBL_DIR, new_base + ".txt")

        cv2.imwrite(new_img_path, out_img)
        write_yolo_label(new_lbl_path, out_bboxes, out_labels)

        made += 1
        aug_made_per_class[c] += 1

    print(f"class {c}: generated {made} / {need} (tries left: {tries_left})")

print("\nAugmentation done.\n")
print("Summary (extra images generated per class):")
for c in all_classes:
    if aug_made_per_class[c] > 0:
        print(f"  class {c}: +{aug_made_per_class[c]}")

print("\nNow your TRAIN split is more balanced (by image-per-class).")
print("valid/test were NOT changed.")