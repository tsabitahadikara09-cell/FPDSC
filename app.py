
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import time
from pathlib import Path
from PIL import Image

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DermaCNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"

CLASSES = ["MEL", "NV", "BKL"]

CLASS_INFO = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BKL": "Benign Keratosis-like Lesions"
}

CLASS_COLORS = {
    "MEL": "#E74C3C",
    "NV": "#3498DB",
    "BKL": "#2ECC71"
}

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #0e0e17;
    color: #f5f5f5;
}

[data-testid="stSidebar"] {
    background-color: #191522;
}

h1, h2, h3 {
    color: #f5f5f5 !important;
}

div[data-testid="stMetric"] {
    background-color: #1c1c28;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #343442;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET LOADER
# =========================================================
@st.cache_data
def load_dataset():

    rows = []

    if not DATASET_DIR.exists():
        return pd.DataFrame(
            columns=["image_id", "label", "path"]
        )

    for file in DATASET_DIR.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp"
        ]:
            continue

        filename = file.stem.upper()

        # Label diambil dari prefix nama file
        label = filename.split("_")[0]

        if label not in CLASSES:
            continue

        try:
            with Image.open(file) as img:
                img.verify()

            rows.append({
                "image_id": file.stem,
                "label": label,
                "path": str(file)
            })

        except Exception:
            continue

    df = pd.DataFrame(
        rows,
        columns=["image_id", "label", "path"]
    )

    return df


# =========================================================
# DATASET UTILITIES
# =========================================================
def get_class_summary(df):

    if df.empty:
        return pd.Series(
            {class_name: 0 for class_name in CLASSES}
        )

    return df["label"].value_counts().reindex(
        CLASSES,
        fill_value=0
    )


def get_sample_image(df, label):

    subset = df[df["label"] == label]

    if subset.empty:
        return None

    row = subset.iloc[0]

    try:
        return Image.open(row["path"]).convert("RGB")
    except Exception:
        return None


def read_image(path):

    try:
        image = Image.open(path).convert("RGB")
        return np.array(image)
    except Exception:
        return None


def get_random_image(df, label):

    subset = df[df["label"] == label]

    if subset.empty:
        return None

    row = subset.sample(1).iloc[0]

    return row


# =========================================================
# INITIAL DATA
# =========================================================
df = load_dataset()

summary = get_class_summary(df)

total_images = len(df)
detected_classes = sum(summary > 0)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🧬 DermaCNN")

    st.caption("Skin Lesion Classification | ISIC 2018")

    st.markdown("---")

    page = st.radio(
        "Navigasi",
        [
            "Home",
            "Dataset",
            "Filtering",
            "Segmentation & Masking",
            "Enhancement",
            "Hair Removal",
            "Grayscale",
            "Training (CNN)",
            "Evaluation"
        ]
    )

    st.markdown("---")

    st.markdown("### Dataset Status")

    if total_images > 0:

        st.success(f"Dataset terbaca: {total_images} gambar")

    else:

        st.error("Dataset belum ditemukan")

    st.caption(
        f"Kelas terdeteksi: {detected_classes} / {len(CLASSES)}"
    )

    if st.button("🔄 Refresh Dataset", use_container_width=True):

        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.caption("DermaCNN | Biomedical Deep Learning Project")
    st.caption("ISIC 2018 • MEL • NV • BKL")


# =========================================================
# HOME
# =========================================================
if page == "Home":

    st.title("🧬 DermaCNN")

    st.markdown(
        """
        ### Skin Lesion Classification

        Aplikasi untuk eksplorasi dataset dan preprocessing citra
        lesi kulit menggunakan pendekatan computer vision dan
        deep learning.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Gambar", total_images)

    col2.metric("Jumlah Kelas", detected_classes)

    col3.metric("Kelas", "MEL • NV • BKL")

    st.markdown("---")

    st.subheader("Distribusi Dataset")

    if total_images > 0:

        st.bar_chart(summary)

    else:

        st.warning(
            "Belum terdapat gambar yang berhasil dibaca dari folder dataset."
        )

    st.subheader("Contoh Gambar per Kelas")

    columns = st.columns(3)

    for i, label in enumerate(CLASSES):

        with columns[i]:

            st.markdown(f"### {label}")

            st.caption(CLASS_INFO[label])

            image = get_sample_image(df, label)

            if image is not None:

                st.image(
                    image,
                    use_container_width=True
                )

            else:

                st.info(f"Belum ada gambar kelas {label}")


# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset":

    st.title("📁 Dataset Explorer")

    st.caption("Dataset ISIC • MEL, NV, BKL")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Gambar", total_images)

    col2.metric("Jumlah Kelas", detected_classes)

    col3.metric(
        "Kelas",
        "MEL • NV • BKL"
    )

    st.markdown("---")

    st.subheader("Distribusi Dataset")

    if total_images > 0:

        chart_df = summary.rename("Jumlah Gambar")

        st.bar_chart(chart_df)

        st.dataframe(
            pd.DataFrame({
                "Kelas": CLASSES,
                "Nama Penyakit": [
                    CLASS_INFO[c] for c in CLASSES
                ],
                "Jumlah Gambar": [
                    int(summary[c]) for c in CLASSES
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Dataset kosong. Pastikan gambar berada langsung di folder dataset/"
        )

    st.markdown("---")

    st.subheader("Contoh Gambar per Kelas")

    cols = st.columns(3)

    for i, label in enumerate(CLASSES):

        with cols[i]:

            st.markdown(f"### {label}")

            image = get_sample_image(df, label)

            if image is not None:

                st.image(
                    image,
                    caption=CLASS_INFO[label],
                    use_container_width=True
                )

            else:

                st.info(f"Gambar kelas {label} belum tersedia")

    st.markdown("---")

    st.subheader("Indeks Dataset")

    if total_images > 0:

        display_df = df[["image_id", "label"]].copy()

        display_df["Nama Kelas"] = display_df["label"].map(
            CLASS_INFO
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FILTERING
# =========================================================
elif page == "Filtering":

    st.title("🧹 Filtering")

    st.write(
        "Mengurangi noise pada citra lesi kulit menggunakan Gaussian Blur."
    )

    label = st.selectbox(
        "Pilih kelas",
        CLASSES
    )

    subset = df[df["label"] == label]

    if subset.empty:

        st.warning(f"Tidak ada gambar pada kelas {label}.")

    else:

        selected = st.selectbox(
            "Pilih gambar",
            subset["image_id"].tolist()
        )

        row = subset[
            subset["image_id"] == selected
        ].iloc[0]

        image = read_image(row["path"])

        if image is not None:

            filtered = cv2.GaussianBlur(
                image,
                (5, 5),
                0
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Filtered")

                st.image(
                    filtered,
                    use_container_width=True
                )


# =========================================================
# SEGMENTATION & MASKING
# =========================================================
elif page == "Segmentation & Masking":

    st.title("🔬 Segmentation & Masking")

    st.write(
        "Membuat mask lesi dengan threshold adaptif dan operasi morfologi."
    )

    label = st.selectbox(
        "Pilih kelas",
        CLASSES
    )

    subset = df[df["label"] == label]

    if subset.empty:

        st.warning(f"Tidak ada gambar pada kelas {label}.")

    else:

        selected = st.selectbox(
            "Pilih gambar",
            subset["image_id"].tolist()
        )

        row = subset[
            subset["image_id"] == selected
        ].iloc[0]

        image = read_image(row["path"])

        if image is not None:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

            blur = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            mask = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51,
                5
            )

            kernel = np.ones(
                (5, 5),
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            segmented = cv2.bitwise_and(
                image,
                image,
                mask=mask
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.subheader("Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Mask")

                st.image(
                    mask,
                    clamp=True,
                )

            with col3:

                st.subheader("Segmented")

                st.image(
                    segmented,
                    use_container_width=True
                )


# =========================================================
# ENHANCEMENT
# =========================================================
elif page == "Enhancement":

    st.title("✨ Image Enhancement")

    st.write(
        "Meningkatkan kontras citra menggunakan CLAHE."
    )

    label = st.selectbox(
        "Pilih kelas",
        CLASSES
    )

    subset = df[df["label"] == label]

    if subset.empty:

        st.warning(f"Tidak ada gambar pada kelas {label}.")

    else:

        selected = st.selectbox(
            "Pilih gambar",
            subset["image_id"].tolist()
        )

        row = subset[
            subset["image_id"] == selected
        ].iloc[0]

        image = read_image(row["path"])

        if image is not None:

            lab = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2LAB
            )

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced_l = clahe.apply(l)

            enhanced_lab = cv2.merge(
                (enhanced_l, a, b)
            )

            enhanced = cv2.cvtColor(
                enhanced_lab,
                cv2.COLOR_LAB2RGB
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Enhanced")

                st.image(
                    enhanced,
                    use_container_width=True
                )


# =========================================================
# HAIR REMOVAL
# =========================================================
elif page == "Hair Removal":

    st.title("🪒 Hair Removal")

    st.write(
        "Menghilangkan rambut pada citra menggunakan Black-Hat "
        "morphology dan inpainting."
    )

    label = st.selectbox(
        "Pilih kelas",
        CLASSES
    )

    subset = df[df["label"] == label]

    if subset.empty:

        st.warning(f"Tidak ada gambar pada kelas {label}.")

    else:

        selected = st.selectbox(
            "Pilih gambar",
            subset["image_id"].tolist()
        )

        row = subset[
            subset["image_id"] == selected
        ].iloc[0]

        image = read_image(row["path"])

        if image is not None:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (17, 17)
            )

            blackhat = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            _, hair_mask = cv2.threshold(
                blackhat,
                10,
                255,
                cv2.THRESH_BINARY
            )

            cleaned = cv2.inpaint(
                image,
                hair_mask,
                3,
                cv2.INPAINT_TELEA
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Hair Removed")

                st.image(
                    cleaned,
                    use_container_width=True
                )


# =========================================================
# GRAYSCALE
# =========================================================
elif page == "Grayscale":

    st.title("⚫ Grayscale")

    st.write(
        "Mengubah citra RGB menjadi grayscale."
    )

    label = st.selectbox(
        "Pilih kelas",
        CLASSES
    )

    subset = df[df["label"] == label]

    if subset.empty:

        st.warning(f"Tidak ada gambar pada kelas {label}.")

    else:

        selected = st.selectbox(
            "Pilih gambar",
            subset["image_id"].tolist()
        )

        row = subset[
            subset["image_id"] == selected
        ].iloc[0]

        image = read_image(row["path"])

        if image is not None:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Grayscale")

                st.image(
                    gray,
                    clamp=True,
                    use_container_width=True
                )


# =========================================================
# TRAINING
# =========================================================
elif page == "Training (CNN)":

    st.title("🧠 Training CNN")

    st.warning(
        "Training membutuhkan jumlah gambar yang memadai. "
        "Dataset dengan hanya beberapa gambar tidak dapat menghasilkan "
        "model klasifikasi yang valid."
    )

    st.write(
        f"Jumlah gambar saat ini: **{total_images}**"
    )

    st.write(
        "Distribusi dataset:"
    )

    st.dataframe(
        summary.rename("Jumlah Gambar"),
        use_container_width=True
    )

    st.info(
        "Modul training CNN akan digunakan setelah dataset "
        "memiliki jumlah data yang cukup."
    )


# =========================================================
# EVALUATION
# =========================================================
elif page == "Evaluation":

    st.title("📊 Evaluation")

    st.warning(
        "Evaluasi model belum dapat dilakukan secara valid "
        "jika dataset hanya berisi sedikit gambar."
    )

    st.write("Distribusi dataset saat ini:")

    st.bar_chart(summary)

    st.info(
        "Tambahkan dataset yang cukup untuk setiap kelas MEL, NV, dan BKL "
        "sebelum melakukan train-test split dan evaluasi model."
    )
