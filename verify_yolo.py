import os
from pathlib import Path

# 🔧 Configure here
dataset_root = r"D:\delete2025\20251205_patent_abnormal\KITCHEN-ASSIST-V2\training\yolo_dataset"

# Expected structure
images_train = Path(dataset_root) / "images" / "train"
images_val   = Path(dataset_root) / "images" / "val"
labels_train = Path(dataset_root) / "labels" / "train"
labels_val   = Path(dataset_root) / "labels" / "val"

print("🔍 Validating YOLOv8 dataset structure...\n")

def check_dir(p, desc):
    if not p.exists():
        print(f"❌ {desc}: NOT FOUND → {p}")
        return False
    if not p.is_dir():
        print(f"❌ {desc}: NOT A DIRECTORY → {p}")
        return False
    print(f"✅ {desc}: OK ({len(list(p.glob('*.*')))} files)")
    return True

def validate_labels(labels_dir, images_dir, split_name):
    if not labels_dir.exists():
        print(f"❌ Labels for '{split_name}' missing: {labels_dir}")
        return False
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_files = [f for f in images_dir.iterdir() if f.suffix.lower() in img_exts]
    txt_files = list(labels_dir.glob("*.txt"))

    img_stems = {f.stem for f in img_files}
    txt_stems = {f.stem for f in txt_files}

    missing_txt = img_stems - txt_stems
    missing_img = txt_stems - img_stems

    print(f"✅ '{split_name}' images: {len(img_files)} | labels: {len(txt_files)}")
    if missing_txt:
        print(f"⚠️  Missing labels for {len(missing_txt)} images in '{split_name}': e.g., {sorted(missing_txt)[:3]}")
    if missing_img:
        print(f"⚠️  Orphaned label files (no matching image) in '{split_name}': {sorted(missing_img)[:3]}")

    # Sample 3 labels to check format
    if txt_files:
        print(f"📝 Sample label check (first 3 .txt files in '{split_name}'):")
        for txt in txt_files[:3]:
            try:
                lines = txt.read_text().strip().splitlines()
                for i, line in enumerate(lines[:2]):  # check up to 2 bboxes
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"❌ {txt.name}: line {i+1} has {len(parts)} values (expected 5)")
                        return False
                    cls, *coords = parts
                    if not cls.isdigit() or int(cls) not in {0, 1}:
                        print(f"❌ {txt.name}: class '{cls}' invalid (must be 0 or 1)")
                        return False
                    if not all(0.0 <= float(c) <= 1.0 for c in coords):
                        print(f"❌ {txt.name}: coords not in [0,1] → {coords}")
                        return False
                print(f"   ✓ {txt.name}: OK ({len(lines)} object(s))")
            except Exception as e:
                print(f"❌ {txt.name}: parse error — {e}")
                return False
    return True

# Run checks
ok = True
ok &= check_dir(images_train, "Train images")
ok &= check_dir(images_val,   "Val images")
ok &= check_dir(labels_train, "Train labels")
ok &= check_dir(labels_val,   "Val labels")

if ok:
    print("\n📊 Detailed validation:")
    ok &= validate_labels(labels_train, images_train, "train")
    ok &= validate_labels(labels_val,   images_val,   "val")

print(f"\n{'🎉 Dataset is YOLOv8-ready!' if ok else '💥 Fix issues above before training.'}")