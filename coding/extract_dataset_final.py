import zipfile
import os
import hashlib
import shutil

# ================= CONFIG ================= #

DATASET_ROOT = "dataset/final"

DATASETS = {
    "bed": [
        "dataset/bed.v1i.yolov8.zip"
    ],
    "chair": [
        "dataset/chair.v1i.yolov8.zip"
    ],
    "coffee_maker": [
        "dataset/coffeemaker.v1i.yolov8.zip"
    ],
    "door": [
        "dataset/door1.v1i.yolov8.zip"
    ],
    "escalator": [
        "dataset/escalator.v1i.yolov8.zip"
    ],
    "fan": [
        "dataset/fan.v1i.yolov8.zip"
    ],
    "fire_extinguisher": [
        "dataset/fire extinguisher.v1i.yolov8.zip",
        "dataset/fire extinguisher1.v1i.yolov8.zip"
    ],
    "laptop": [
        "dataset/laptop.v1i.yolov8.zip",
        "dataset/laptop1.v1i.yolov8.zip"
    ],
    "microwave": [
        "dataset/microwave.v1i.yolov8.zip"
    ],
    "refrigerator": [
        "dataset/refrigerator.v1i.yolov8.zip"
    ],
    "rice_cooker": [
        "dataset/ricecooker.v1i.yolov8.zip"
    ],
    "sofa": [
        "dataset/Sofa.v1i.yolov8.zip"
    ],
    "stair": [
        "dataset/stair.v1i.yolov8.zip"
    ],
    "table": [
        "dataset/table.v1i.yolov8.zip"
    ],
    "toilet": [
        "dataset/toilet.v1i.yolov8.zip"
    ],
    "tv": [
        "dataset/tv.v1i.yolov8.zip",
        "dataset/tv2.v1i.yolov8.zip"
    ],
    "washing_machine": [
        "dataset/Washing Machine.v1i.yolov8.zip"
    ],
    "water_dispenser": [
        "dataset/wd.v1i.yolov8.zip"
    ],
    "wheelchair": [
        "dataset/Wheelchair.v1i.yolov8.zip"
    ]
}

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# ================= CORE FUNCTION ================= #

def extract_zip(zip_path, class_output_dir):
    filename_mapping = {}

    print(f"\nExtracting: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        # ---------- PASS 1: collect files ----------
        all_files = [
            m.filename for m in zip_ref.infolist()
            if not m.filename.endswith('/')
        ]

        print(f"Found {len(all_files)} files")

        # ---------- GROUP BY BASE NAME ----------
        file_groups = {}
        for filename in all_files:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            file_groups.setdefault(base_name, []).append(filename)

        # ---------- HASH RENAMING ----------
        for base_name, files in file_groups.items():
            name_hash = hashlib.md5(base_name.encode()).hexdigest()[:12]
            for original_file in files:
                ext = os.path.splitext(original_file)[1]
                filename_mapping[original_file] = f"{name_hash}{ext}"

        # ---------- PASS 2: EXTRACT ----------
        for member in zip_ref.infolist():
            if member.filename.endswith('/'):
                continue

            new_filename = filename_mapping.get(
                member.filename,
                hashlib.md5(member.filename.encode()).hexdigest()[:12]
                + os.path.splitext(member.filename)[1]
            )

            lower_name = member.filename.lower()

            # dataset split
            if 'train' in lower_name:
                split = 'train'
            elif 'valid' in lower_name or 'val' in lower_name:
                split = 'valid'
            elif 'test' in lower_name:
                split = 'test'
            else:
                continue

            # file type
            if member.filename.lower().endswith(IMAGE_EXTS):
                ftype = 'images'
            elif member.filename.lower().endswith('.txt'):
                ftype = 'labels'
            else:
                continue

            target_dir = os.path.join(class_output_dir, split, ftype)
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, new_filename)

            with zip_ref.open(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

    print(f"Done: {zip_path}")

# ================= MAIN ================= #

# Clear dataset root ONCE
if os.path.exists(DATASET_ROOT):
    shutil.rmtree(DATASET_ROOT)
os.makedirs(DATASET_ROOT, exist_ok=True)

# Process each class
for class_name, zip_list in DATASETS.items():
    print(f"\nProcessing class: {class_name}")
    class_dir = os.path.join(DATASET_ROOT, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for zip_path in zip_list:
        extract_zip(zip_path, class_dir)

# ================= VERIFY ================= #

print("\nVerifying label-image correspondence...")

for class_name in DATASETS.keys():
    print(f"\nClass: {class_name}")
    class_dir = os.path.join(DATASET_ROOT, class_name)

    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(class_dir, split, 'images')
        labels_dir = os.path.join(class_dir, split, 'labels')

        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            images = {
                os.path.splitext(f)[0] for f in os.listdir(images_dir)
                if f.lower().endswith(IMAGE_EXTS)
            }
            labels = {
                os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                if f.lower().endswith('.txt')
            }

            print(
                f"  {split}: {len(images)} images, "
                f"{len(labels)} labels, "
                f"{len(images & labels)} matched"
            )

print("\nAll datasets extracted successfully!")
print(f"Dataset root: {DATASET_ROOT}")