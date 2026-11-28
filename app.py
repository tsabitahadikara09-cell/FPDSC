import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from modules.dataset import (
    load_dataset_index,
    class_summary,
    get_class_images,
    get_image_by_id,
    get_image_options
)

from modules.preprocessing import (
    mean_filter,
    median_filter,
    gaussian_filter,
    otsu_mask,
    apply_segmentation,
    enhance_image,
    hair_removal,
    grayscale_image
)

from modules.training import train_model
from modules.evaluation import evaluate_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DermaCNN",
    page_icon="🧬",
    layout="wide"
)


# =========================================================
# DATASET
# =========================================================

@st.cache_data
def get_dataset():

    df, error = load_dataset_index()

    return df, error


df, dataset_error = get_dataset()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧬 DermaCNN")
st.sidebar.caption("Skin Lesion Classification | ISIC 2018")

page = st.sidebar.radio(
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

st.sidebar.divider()

if df is not None and not df.empty:

    st.sidebar.success(
        f"Dataset terbaca: {len(df)} gambar"
    )

    summary = class_summary(df)

    detected = sum(
        1 for value in summary.values()
        if value > 0
    )

    st.sidebar.caption(
        f"Kelas terdeteksi: {detected}/3"
    )

else:

    st.sidebar.warning("Dataset belum terbaca.")

if st.sidebar.button("🔄 Refresh Dataset"):

    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    "DermaCNN | Biomedical Deep Learning Project"
)

st.sidebar.caption(
    "ISIC 2018 • MEL • NV • BKL"
)


# =========================================================
# HELPER
# =========================================================

def choose_image(label):

    subset = get_class_images(df, label)

    if subset.empty:
        st.warning(f"Tidak ada gambar pada kelas {label}.")
        return None, None

    options = get_image_options(df, label)

    selected = st.selectbox(
        "Pilih gambar",
        options,
        key=f"image_{label}_{page}"
    )

    image, actual_label = get_image_by_id(
        df,
        selected
    )

    return image, selected


def show_image(image, title):

    st.subheader(title)

    st.image(
        image,
        use_container_width=True
    )


def plot_distribution(summary):

    fig, ax = plt.subplots(figsize=(8, 4))

    labels = list(summary.keys())
    values = list(summary.values())

    ax.bar(labels, values)

    ax.set_title("Distribusi Dataset")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Gambar")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("🧬 DermaCNN")

    st.markdown(
        """
        ### Skin Lesion Classification

        Aplikasi preprocessing dan klasifikasi citra dermoskopi
        berbasis Deep Learning.

        Dataset terdiri dari tiga kelas:

        - **MEL** — Melanoma
        - **NV** — Melanocytic Nevi
        - **BKL** — Benign Keratosis

        """
    )

    if df is not None and not df.empty:

        summary = class_summary(df)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Gambar",
            len(df)
        )

        col2.metric(
            "Jumlah Kelas",
            len([x for x in summary.values() if x > 0])
        )

        col3.metric(
            "Kategori",
            "MEL • NV • BKL"
        )

        st.subheader("Distribusi Dataset")

        plot_distribution(summary)

    else:

        st.error(dataset_error)


# =========================================================
# DATASET
# =========================================================

elif page == "Dataset":

    st.title("📁 Dataset Explorer")

    if df.empty:

        st.warning(dataset_error)

    else:

        summary = class_summary(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Gambar", len(df))

        col2.metric(
            "Jumlah Kelas",
            sum(value > 0 for value in summary.values())
        )

        col3.metric(
            "Kelas",
            "MEL • NV • BKL"
        )

        st.subheader("Distribusi Dataset")

        plot_distribution(summary)

        st.subheader("Contoh Gambar per Kelas")

        cols = st.columns(3)

        for i, label in enumerate(["MEL", "NV", "BKL"]):

            with cols[i]:

                st.markdown(f"**{label}**")

                image, image_id = sample_image(
                    df, label
                ) if False else (None, None)

                subset = get_class_images(df, label)

                if not subset.empty:

                    first = subset.iloc[0]

                    try:

                        img = Image.open(
                            first["path"]
                        ).convert("RGB")

                        st.image(
                            img,
                            use_container_width=True
                        )

                        st.caption(first["image_id"])

                    except Exception:

                        st.warning("Gambar gagal dibaca.")

                else:

                    st.info("Belum ada gambar.")


# =========================================================
# FILTERING
# =========================================================

elif page == "Filtering":

    st.title("🔧 Image Filtering")

    st.write(
        "Filtering digunakan untuk mengurangi noise "
        "pada citra dermoskopi."
    )

    label = st.selectbox(
        "Pilih kelas",
        ["MEL", "NV", "BKL"]
    )

    image, image_id = choose_image(label)

    if image is not None:

        method = st.selectbox(
            "Metode filtering",
            [
                "Mean Filter",
                "Median Filter",
                "Gaussian Filter"
            ]
        )

        kernel_size = st.slider(
            "Kernel size",
            min_value=3,
            max_value=15,
            step=2,
            value=5
        )

        if method == "Mean Filter":

            processed = mean_filter(
                image,
                kernel_size
            )

        elif method == "Median Filter":

            processed = median_filter(
                image,
                kernel_size
            )

        else:

            processed = gaussian_filter(
                image,
                kernel_size
            )

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(processed, method)


# =========================================================
# SEGMENTATION & MASKING
# =========================================================

elif page == "Segmentation & Masking":

    st.title("🔬 Segmentation & Masking")

    st.write(
        "Mask lesi dibuat menggunakan Otsu Thresholding "
        "dan operasi morfologi untuk membersihkan noise."
    )

    label = st.selectbox(
        "Pilih kelas",
        ["MEL", "NV", "BKL"]
    )

    image, image_id = choose_image(label)

    if image is not None:

        mask = otsu_mask(image)

        segmented = apply_segmentation(
            image,
            mask
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(mask, "Mask (Otsu)")

        with col3:
            show_image(segmented, "Segmented")


# =========================================================
# ENHANCEMENT
# =========================================================

elif page == "Enhancement":

    st.title("✨ Image Enhancement")

    st.write(
        "Enhancement meningkatkan kontras lokal "
        untuk membantu visualisasi lesi."
    )

    label = st.selectbox(
        "Pilih kelas",
        ["MEL", "NV", "BKL"]
    )

    image, image_id = choose_image(label)

    if image is not None:

        clip_limit = st.slider(
            "CLAHE Clip Limit",
            1.0,
            5.0,
            2.0
        )

        processed = enhance_image(
            image,
            clip_limit=clip_limit
        )

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(processed, "Enhanced")


# =========================================================
# HAIR REMOVAL
# =========================================================

elif page == "Hair Removal":

    st.title("🧹 Hair Removal")

    st.write(
        "Mengurangi rambut pada citra dermoskopi "
        menggunakan Black-hat Morphology dan Inpainting."
    )

    label = st.selectbox(
        "Pilih kelas",
        ["MEL", "NV", "BKL"]
    )

    image, image_id = choose_image(label)

    if image is not None:

        processed = hair_removal(image)

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(processed, "Hair Removed")


# =========================================================
# GRAYSCALE
# =========================================================

elif page == "Grayscale":

    st.title("⚫ Grayscale")

    label = st.selectbox(
        "Pilih kelas",
        ["MEL", "NV", "BKL"]
    )

    image, image_id = choose_image(label)

    if image is not None:

        processed = grayscale_image(image)

        col1, col2 = st.columns(2)

        with col1:
            show_image(image, "Original")

        with col2:
            show_image(processed, "Grayscale")


# =========================================================
# TRAINING
# =========================================================

elif page == "Training (CNN)":

    st.title("🧠 Training CNN")

    st.write(
        "Model CNN digunakan untuk klasifikasi "
        "citra lesi kulit ke dalam tiga kategori."
    )

    if df.empty:

        st.warning("Dataset kosong.")

    else:

        summary = class_summary(df)

        st.subheader("Distribusi Dataset")

        st.dataframe(
            pd.DataFrame({
                "Label": list(summary.keys()),
                "Jumlah Gambar": list(summary.values())
            }),
            use_container_width=True
        )

        st.warning(
            "Training memerlukan jumlah gambar yang memadai "
            "pada setiap kelas. Dataset yang terlalu sedikit "
            "tidak dapat menghasilkan evaluasi yang valid."
        )

        epochs = st.slider(
            "Epoch",
            min_value=1,
            max_value=50,
            value=10
        )

        if st.button("🚀 Mulai Training"):

            try:

                with st.spinner("Training CNN berlangsung..."):

                    model, history, encoder = train_model(
                        df,
                        epochs=epochs
                    )

                st.session_state["model"] = model
                st.session_state["history"] = history
                st.session_state["encoder"] = encoder

                st.success("Training selesai.")

                history_df = pd.DataFrame(
                    history.history
                )

                st.line_chart(
                    history_df[["accuracy", "val_accuracy"]]
                )

                st.line_chart(
                    history_df[["loss", "val_loss"]]
                )

            except Exception as e:

                st.error(
                    f"Training gagal: {e}"
                )


# =========================================================
# EVALUATION
# =========================================================

elif page == "Evaluation":

    st.title("📊 Evaluation")

    summary = class_summary(df)

    if df.empty:

        st.warning("Dataset kosong.")

    else:

        st.subheader("Distribusi Dataset")

        plot_distribution(summary)

        st.warning(
            "Evaluasi model membutuhkan dataset yang cukup "
            "dan data test terpisah dari data training. "
            "Dengan hanya beberapa gambar, hasil evaluasi "
            "tidak merepresentasikan performa model sebenarnya."
        )

        if "model" not in st.session_state:

            st.info(
                "Silakan lakukan training terlebih dahulu."
            )

        else:

            st.success(
                "Model training tersedia pada session ini."
            )

            st.info(
                "Gunakan data test terpisah untuk evaluasi "
                "yang valid."
            )
