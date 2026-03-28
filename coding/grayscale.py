import os
import cv2

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/yolo_all2"
SPLIT = "train" 
IMG_DIR = os.path.join(DATA_ROOT, SPLIT, "images")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Set True to overwrite images
# Set False to save into a new folder (recommended first run)
OVERWRITE = True

OUT_DIR = IMG_DIR if OVERWRITE else IMG_DIR + "_gray"

if not OVERWRITE:
    os.makedirs(OUT_DIR, exist_ok=True)

print(f"Converting images to grayscale (3-channel)")
print(f"Source: {IMG_DIR}")
print(f"Output: {OUT_DIR}")
print()

count = 0

for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith(IMAGE_EXTS):
        continue

    src_path = os.path.join(IMG_DIR, fname)
    dst_path = os.path.join(OUT_DIR, fname)

    img = cv2.imread(src_path)
    if img is None:
        continue

    # ---- Convert to grayscale ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- Convert back to 3-channel (YOLO safe) ----
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(dst_path, gray_3ch)
    count += 1

print(f"Done. {count} images converted to grayscale (3-channel).")
print("Labels were NOT modified.")