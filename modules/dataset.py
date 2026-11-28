
from pathlib import Path
import pandas as pd
from PIL import Image

# =========================================================
# PATH CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"

CLASSES = ["MEL", "NV", "BKL"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# LABEL DETECTION FROM FILENAME
# =========================================================

def get_label_from_filename(filename):
    """
    Menentukan kelas berdasarkan prefix nama file.

    Contoh:
    MEL_ISIC_0024310.jpg -> MEL
    NV_ISIC_0024306.jpg   -> NV
    BKL_ISIC_0024312.jpg -> BKL
    """

    filename = Path(filename).name.upper()

    for label in CLASSES:
        if filename.startswith(label + "_"):
            return label

    return "Unlabeled"


# =========================================================
# LOAD DATASET INDEX
# =========================================================

def load_dataset_index():

    rows = []

    if not DATASET_DIR.exists():
        return pd.DataFrame(), (
            f"Folder dataset tidak ditemukan: {DATASET_DIR}"
        )

    # Membaca seluruh gambar di folder dataset
    # dan subfoldernya, kecuali folder images
    image_files = []

    for path in DATASET_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Abaikan folder images jika memang kosong
        # atau digunakan untuk kebutuhan lain
        if "images" in path.parts:
            continue

        image_files.append(path)

    if not image_files:
        return pd.DataFrame(), (
            "Tidak ditemukan gambar dalam folder dataset."
        )

    # =====================================================
    # PROCESS EACH IMAGE
    # =====================================================

    for image_path in sorted(image_files):

        # Label langsung dari nama file
        label = get_label_from_filename(image_path.name)

        # Abaikan file yang tidak memiliki label
        if label not in CLASSES:
            continue

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

    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    df = pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "label",
            "path"
        ]
    )

    if df.empty:
        return df, (
            "Tidak ditemukan gambar valid dengan label MEL, NV, atau BKL. "
            "Pastikan nama file diawali dengan MEL_, NV_, atau BKL_."
        )

    return df, None


# =========================================================
# CLASS SUMMARY
# =========================================================

def class_summary(df):

    if df is None or df.empty:
        return {
            label: 0
            for label in CLASSES
        }

    return {
        label: int(
            (df["label"] == label).sum()
        )
        for label in CLASSES
    }


# =========================================================
# SAMPLE IMAGE
# =========================================================

def sample_image(df, label):

    if df is None or df.empty:
        return None, None

    subset = df[
        df["label"] == label
    ]

    if subset.empty:
        return None, None

    row = subset.iloc[0]

    try:

        image = Image.open(
            row["path"]
        ).convert("RGB")

        return image, row["image_id"]

    except Exception:
        return None, None


# =========================================================
# GET ALL IMAGES FROM CLASS
# =========================================================

def get_images_by_class(df, label):

    if df is None or df.empty:
        return pd.DataFrame()

    return df[
        df["label"] == label
    ].reset_index(drop=True)


# =========================================================
# GET RANDOM SAMPLE
# =========================================================

def get_random_samples(df, label, n=5, random_state=42):

    if df is None or df.empty:
        return pd.DataFrame()

    subset = df[
        df["label"] == label
    ]

    if subset.empty:
        return pd.DataFrame()

    n = min(n, len(subset))

    return subset.sample(
        n=n,
        random_state=random_state
    ).reset_index(drop=True)


# =========================================================
# DATASET INFORMATION
# =========================================================

def dataset_info(df):

    if df is None or df.empty:

        return {
            "total_images": 0,
            "total_classes": 0,
            "class_distribution": class_summary(df)
        }

    summary = class_summary(df)

    detected_classes = sum(
        count > 0
        for count in summary.values()
    )

    return {
        "total_images": len(df),
        "total_classes": detected_classes,
        "class_distribution": summary
    }
