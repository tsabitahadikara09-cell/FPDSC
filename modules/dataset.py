from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

CLASSES = ["MEL", "NV", "BKL"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_label_from_filename(filename):
    """
    Label diambil dari prefix nama file:
    MEL_ISIC_xxx.jpg
    NV_ISIC_xxx.jpg
    BKL_ISIC_xxx.jpg
    """

    name = Path(filename).stem.upper()

    for label in CLASSES:
        if name.startswith(label + "_") or name == label:
            return label

    return None


def load_dataset_index():
    rows = []

    if not DATASET_DIR.exists():
        return pd.DataFrame(
            columns=["image_id", "label", "path"]
        ), f"Folder dataset tidak ditemukan: {DATASET_DIR}"

    image_files = sorted(
        [
            p for p in DATASET_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    for image_path in image_files:

        label = get_label_from_filename(image_path.name)

        # Hanya mengambil gambar dengan label MEL, NV, BKL
        if label not in CLASSES:
            continue

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
        return df, (
            "Tidak ditemukan gambar valid. "
            "Pastikan nama file menggunakan prefix MEL_, NV_, atau BKL_."
        )

    return df, None


def class_summary(df):

    if df is None or df.empty:
        return {label: 0 for label in CLASSES}

    return {
        label: int((df["label"] == label).sum())
        for label in CLASSES
    }


def get_class_images(df, label):

    if df is None or df.empty:
        return pd.DataFrame()

    return df[df["label"] == label].reset_index(drop=True)


def sample_image(df, label):

    subset = get_class_images(df, label)

    if subset.empty:
        return None, None

    row = subset.iloc[0]

    try:
        image = Image.open(row["path"]).convert("RGB")
        return image, row["image_id"]

    except Exception:
        return None, None


def get_image_by_id(df, image_id):

    if df is None or df.empty:
        return None, None

    subset = df[df["image_id"] == image_id]

    if subset.empty:
        return None, None

    row = subset.iloc[0]

    try:
        image = Image.open(row["path"]).convert("RGB")
        return image, row["label"]

    except Exception:
        return None, None


def get_image_options(df, label):

    subset = get_class_images(df, label)

    if subset.empty:
        return []

    return subset["image_id"].tolist()
