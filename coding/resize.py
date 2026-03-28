import os
import cv2

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/yolo_all2"
SPLITS = ["train", "valid", "test"]

TARGET_SIZE = 640

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

print("Resizing dataset images to 640x640\n")

total = 0

for split in SPLITS:

    IMG_DIR = os.path.join(DATA_ROOT, split, "images")

    print(f"Processing split: {split}")
    print(f"{IMG_DIR}")

    for fname in os.listdir(IMG_DIR):

        if not fname.lower().endswith(IMAGE_EXTS):
            continue

        path = os.path.join(IMG_DIR, fname)

        img = cv2.imread(path)
        if img is None:
            continue

        # Resize image
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # Overwrite image
        cv2.imwrite(path, resized)

        total += 1

    print(f"{split} done\n")

print(f"Resize complete. {total} images resized to {TARGET_SIZE}x{TARGET_SIZE}")
print("YOLO labels unchanged because they are normalized.")