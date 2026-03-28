from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG (edit these)
# ----------------------------
DATA_YAML = "dataset/yolo_all2/data.yaml" 
PRETRAINED = "yolov8n.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 6
DEVICE = 0  # set "cpu" if no GPU

PROJECT_DIR = "runs"
RUN_NAME = "indoor_objects_fixed_aug_gray_11"

# Keep EXACT settings you requested
TRAIN_KWARGS = dict(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    optimizer="AdamW",
    momentum=0.9,
    weight_decay=0.01,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.5,
    amp=False,
    workers=0,
    cache=False,
    deterministic=False,
    val=True,
    save=True,
    save_period=5,
    plots=True,
    patience=50,
    project=PROJECT_DIR,
    name=RUN_NAME,
    verbose=True,
)

# ----------------------------
# Helpers: Plot training curves
# ----------------------------
def plot_training_curves(results_csv: Path, out_dir: Path):
    if not results_csv.exists():
        print(f"results.csv not found at: {results_csv}")
        return

    df = pd.read_csv(results_csv)

    # Try common YOLOv8 columns (varies by version)
    candidates = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Cls Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
        ("metrics/mAP50(B)", "mAP50 (box)"),
        ("metrics/mAP50-95(B)", "mAP50-95 (box)"),
        ("metrics/precision(B)", "Precision (box)"),
        ("metrics/recall(B)", "Recall (box)"),
    ]

    # Plot each metric that exists
    for col, title in candidates:
        if col not in df.columns:
            continue

        plt.figure()
        plt.plot(df[col])
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(col)
        out_path = out_dir / f"{col.replace('/', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Training curves saved into: {out_dir}")

# ----------------------------
# Helpers: Per-class metrics
# ----------------------------
def per_class_report(val_metrics, out_dir: Path):
    """
    Save per-class + overall metrics into CSV and TXT.

    Column mapping used for FYP table:
    - mAP     = AP50_95
    - AP IoU  = AP50
    - J       = Precision
    - F       = Recall
    - J&R     = (Precision + Recall) / 2
    """

    names = val_metrics.names
    rows = []

    box = getattr(val_metrics, "box", None)

    maps = getattr(box, "maps", None) if box else None       # per-class AP50-95
    ap50s = getattr(box, "ap50", None) if box else None   # per-class AP50
    p = getattr(box, "p", None) if box else None             # per-class precision
    r = getattr(box, "r", None) if box else None             # per-class recall

    if maps is None:
        print("Could not extract per-class arrays from YOLO validation results.")

        overall_precision = float(sum(p) / len(p)) if p is not None and len(p) > 0 else None
        overall_recall = float(sum(r) / len(r)) if r is not None and len(r) > 0 else None
        overall_jr = (
            (overall_precision + overall_recall) / 2
            if overall_precision is not None and overall_recall is not None
            else None
        )

        overall = {
            "class_id": -1,
            "class_name": "OVERALL",
            "mAP": float(getattr(box, "map", -1.0)) if box else -1.0,
            "AP_IoU": float(getattr(box, "map50", -1.0)) if box else -1.0,
            "J": overall_precision,
            "F": overall_recall,
            "J&R": overall_jr,
        }

        df = pd.DataFrame([overall])
        df.to_csv(out_dir / "per_class_metrics.csv", index=False)

        print("\nOverall metrics only:")
        print(df.to_string(index=False))

        with open(out_dir / "per_class_metrics.txt", "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
        return

    num_classes = len(names)

    for cid in range(num_classes):
        cname = names[cid] if isinstance(names, dict) else str(cid)

        precision = float(p[cid]) if (p is not None and cid < len(p)) else None
        recall = float(r[cid]) if (r is not None and cid < len(r)) else None
        jr = ((precision + recall) / 2) if (precision is not None and recall is not None) else None

        row = {
            "class_id": cid,
            "class_name": cname,
            "mAP": float(maps[cid]) if cid < len(maps) else None,
            "AP_IoU": float(ap50s[cid]) if (ap50s is not None and cid < len(ap50s)) else None,
            "J": precision,
            "F": recall,
            "J&R": jr,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("class_id", ascending=True)

    # ---------- overall row ----------
    overall_precision = float(sum([x for x in p if x is not None]) / len(p)) if p is not None and len(p) > 0 else None
    overall_recall = float(sum([x for x in r if x is not None]) / len(r)) if r is not None and len(r) > 0 else None
    overall_jr = (
        (overall_precision + overall_recall) / 2
        if overall_precision is not None and overall_recall is not None
        else None
    )

    overall_row = {
        "class_id": -1,
        "class_name": "OVERALL",
        "mAP": float(getattr(box, "map", None)) if box else None,
        "AP_IoU": float(getattr(box, "map50", None)) if box else None,
        "J": overall_precision,
        "F": overall_recall,
        "J&R": overall_jr,
    }

    df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

    # save csv
    df.to_csv(out_dir / "per_class_metrics.csv", index=False)

    # print full table
    show_cols = ["class_id", "class_name", "mAP", "AP_IoU", "J", "F", "J&R"]
    print("\nPer-class metrics + overall:")
    print(df[show_cols].to_string(index=False))

    # save txt
    report_txt = out_dir / "per_class_metrics.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(df[show_cols].to_string(index=False))

    print(f"\nSaved CSV: {out_dir / 'per_class_metrics.csv'}")
    print(f"Saved TXT: {report_txt}")

# ----------------------------
# MAIN
# ----------------------------
def train_and_eda():
    # 1) Train
    model = YOLO(PRETRAINED)

    train_res = model.train(**TRAIN_KWARGS)

    # 2) Locate run directory (Ultralytics returns: runs/detect/<name>)
    save_dir = Path(getattr(train_res, "save_dir", Path(PROJECT_DIR) / "detect" / RUN_NAME))
    print(f"\nRun directory: {save_dir}")

    # 3) Find weights
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    print(f"Expecting weights at: {weights_dir}")

    if best_pt.exists():
        print(f"Found best.pt: {best_pt}")
        weights_path = best_pt
    elif last_pt.exists():
        print(f"best.pt not found, using last.pt: {last_pt}")
        weights_path = last_pt
    else:
        raise FileNotFoundError(
            f"No weights found. Checked:\n- {best_pt}\n- {last_pt}\n"
            "Training may not have started, crashed early, or the run directory is different."
        )

    # 4) Load model
    best_model = YOLO(str(weights_path))

    # run validation again to ensure we have metrics object
    val_res = best_model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        device=DEVICE,
        plots=True,     # confusion matrix, PR curve, etc (Ultralytics built-in)
        save_json=True  # saves COCO-like json in run dir
    )

    # 5) Overall metrics
    box = getattr(val_res, "box", None)
    if box:
        print("\nOverall Validation Metrics:")
        print(f"   mAP50-95: {box.map:.4f}")
        print(f"   mAP50:    {box.map50:.4f}")
        print(f"   mAP75:    {box.map75:.4f}")

    # 6) Per-class report CSV
    per_class_report(val_res, save_dir)

    print("\nTraining + EDA completed.")

if __name__ == "__main__":
    train_and_eda()