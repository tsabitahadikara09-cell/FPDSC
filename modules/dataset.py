
from pathlib import Path
import pandas as pd
from PIL import Image

# =========================
# PATH CONFIGURATION
# =========================
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGE_DIR = DATASET_DIR

# CSV label jika tersedia
CSV_PATH = DATASET_DIR / "metadata.csv"

CLASSES = ["MEL", "NV", "BKL"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# =========================
# LOAD DATASET INDEX
# =========================
def load_dataset_index():

    rows = []

    if not DATASET_DIR.exists():
        return pd.DataFrame(), f"Folder dataset tidak ditemukan: {DATASET_DIR}"

    # Cari gambar secara recursive
    image_files = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        return pd.DataFrame(), "Tidak ditemukan gambar di folder dataset."

    for image_path in sorted(image_files):

        # Tentukan label berdasarkan nama folder
        label = None

        for parent in image_path.parents:
            if parent.name.upper() in CLASSES:
                label = parent.name.upper()
                break

        # Jika tidak ditemukan label folder,
        # coba gunakan CSV
        if label is None and CSV_PATH.exists():

            try:
                metadata = pd.read_csv(CSV_PATH)

                image_id = image_path.stem

                if "image_id" in metadata.columns:
                    matched = metadata[
                        metadata["image_id"] == image_id
                    ]

                    if not matched.empty:
                        label_column = next(
                            (
                                c for c in CLASSES
                                if c in metadata.columns
                            ),
                            None
                        )

                        if label_column:
                            label = label_column

            except Exception:
                pass

        # Gambar tanpa label tetap dapat dibaca
        if label is None:
            label = "Unlabeled"

        # Validasi gambar
        try:
            with Image.open(image_path) as img:
                img.verify()

            rows.append({
                "image_id": image_path.stem,
                "label": label,
                "path": str(image_path)
            })

        except Exception:
            continue

    df = pd.DataFrame(
        rows,
        columns=["image_id", "label", "path"]
    )

    if df.empty:
        return df, "Tidak ada gambar valid yang dapat dibaca."

    return df, None


# =========================
# CLASS SUMMARY
# =========================
def class_summary(df):

    if df is None or df.empty:
        return {c: 0 for c in CLASSES}

    return {
        c: int((df["label"] == c).sum())
        for c in CLASSES
    }


# =========================
# SAMPLE IMAGE
# =========================
def sample_image(df, label):

    if df is None or df.empty:
        return None, None

    subset = df[df["label"] == label]

    if subset.empty:
        return None, None

    row = subset.iloc[0]

    try:
        image = Image.open(row["path"]).convert("RGB")
        return image, row["image_id"]

    except Exception:
        return None, None
