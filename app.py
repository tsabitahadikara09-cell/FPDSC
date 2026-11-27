import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from modules.dataset import load_dataset_index, class_summary, sample_image
from modules.preprocessing import preprocess_image, METHOD_DESCRIPTIONS
from modules.training import train_model, MODEL_PATH
from modules.evaluation import evaluate_model

st.set_page_config(page_title="DermaCNN | ISIC 2018", page_icon="🧬", layout="wide")

st.markdown("""<style>
.stApp{background:#111019;color:#f3efff}.block-container{padding-top:1.25rem;padding-bottom:2rem}
[data-testid='stSidebar']{background:#191521}.metric-card{background:#211d2b;border:1px solid #393243;padding:18px;border-radius:14px}
div.stButton>button{border-radius:10px}h1,h2,h3{color:#f4efff}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset_index()

df, dataset_error = get_data()
counts = class_summary(df) if not df.empty else {"MEL":0,"NV":0,"BKL":0}


def show_image(image, caption=None):
    if image is not None:
        st.image(image, caption=caption, use_container_width=True)


def sidebar():
    st.sidebar.markdown("# 🧬 DermaCNN")
    st.sidebar.caption("Skin Lesion Classification | ISIC 2018")
    pages = ["Home", "Dataset", "Filtering", "Segmentation & Masking", "Enhancement", "Hair Removal", "Grayscale", "Training (CNN)", "Evaluation"]
    page = st.sidebar.radio("Navigasi", pages)
    st.sidebar.divider()
    st.sidebar.markdown("### Dataset Status")
    if dataset_error:
        st.sidebar.warning("Dataset belum terbaca")
        st.sidebar.caption(dataset_error)
    else:
        st.sidebar.success(f"Dataset terbaca: {len(df):,} gambar")
        st.sidebar.caption(f"Kelas terdeteksi: {df.label.nunique()} / 3")
    if st.sidebar.button("🔄 Refresh Dataset", use_container_width=True):
        get_data.clear(); st.rerun()
    st.sidebar.divider()
    st.sidebar.caption("DermaCNN | Biomedical Deep Learning Project")
    st.sidebar.caption("ISIC 2018 • MEL • NV • BKL")
    return page


page = sidebar()


def page_home():
    st.title("🧬 DermaCNN")
    st.subheader("Skin Lesion Classification Dashboard")
    st.write("Demo pengolahan citra lesi kulit dan klasifikasi tiga kelas menggunakan CNN.")
    if dataset_error:
        st.error(dataset_error); return
    a,b,c,d = st.columns(4)
    a.metric("Total Images", len(df)); b.metric("Total Classes", df.label.nunique())
    c.metric("MEL", counts.get("MEL",0)); d.metric("NV", counts.get("NV",0))
    st.markdown("### Distribusi Kelas")
    fig, ax = plt.subplots(figsize=(8,3.8)); ax.bar(counts.keys(), counts.values()); ax.set_ylabel("Jumlah gambar"); ax.set_title("Dataset Mini — MEL, NV, BKL")
    st.pyplot(fig); plt.close(fig)
    st.markdown("### Sampel Citra")
    cols=st.columns(3)
    for col,label in zip(cols,["MEL","NV","BKL"]):
        img,imgid=sample_image(df,label)
        with col: st.markdown(f"**{label}**"); show_image(img,imgid)
    st.info("Dataset saat ini adalah dataset mini untuk demonstrasi alur aplikasi, bukan dataset yang memadai untuk validasi klinis atau penelitian performa model.")


def page_dataset():
    st.title("📁 Dataset Explorer")
    st.caption("Dataset mini ISIC — MEL, NV, BKL")
    if dataset_error:
        st.error(dataset_error); return
    a,b,c=st.columns(3); a.metric("Total gambar",len(df)); b.metric("Jumlah kelas",df.label.nunique()); c.metric("Kelas", "MEL • NV • BKL")
    st.subheader("Distribusi Dataset")
    chart=pd.DataFrame({"Kelas":list(counts.keys()),"Jumlah":list(counts.values())})
    st.bar_chart(chart.set_index("Kelas"))
    st.subheader("Contoh gambar per kelas")
    cols=st.columns(3)
    for col,label in zip(cols,["MEL","NV","BKL"]):
        img,imgid=sample_image(df,label)
        with col: st.markdown(f"**{label}**"); show_image(img,imgid)
    with st.expander("Indeks Dataset"):
        st.dataframe(df, use_container_width=True)


def page_preprocess(method):
    st.title(method)
    st.write(METHOD_DESCRIPTIONS.get(method,"Pemrosesan citra."))
    if dataset_error:
        st.error(dataset_error); return
    label=st.selectbox("Pilih kelas",["MEL","NV","BKL"],key=f"label_{method}")
    subset=df[df.label==label].reset_index(drop=True)
    if subset.empty: st.warning("Tidak ada gambar pada kelas ini."); return
    idx=st.slider("Pilih gambar",0,len(subset)-1,0,key=f"idx_{method}")
    row=subset.iloc[idx]
    original=Image.open(row.path).convert("RGB")
    kwargs={}
    if method=="Filtering":
        ftype=st.selectbox("Jenis filter",["Median","Gaussian"])
        kernel=st.slider("Kernel size",3,15,5,step=2)
        kwargs.update(filter_type=ftype,kernel=kernel)
    elif method=="Enhancement":
        kwargs["clip_limit"]=st.slider("Contrast limit",1.0,5.0,2.0,0.5)
    result=preprocess_image(original,method,**kwargs)
    st.caption(f"Image ID: {row.image_id} • Class: {row.label}")
    if method=="Segmentation & Masking":
        # Show only the manually selected lesion region and its generated mask.
        st.markdown("#### Area lesi yang dipilih")
        st.caption("Gambar asli dengan area ROI yang dapat dipilih, lalu hasil segmentasi dan mask.")
        roi=st.checkbox("Aktifkan area ROI manual",value=True)
        if roi:
            st.warning("Untuk versi ini, ROI manual berupa pemilihan area persegi belum diaktifkan. Hasil yang ditampilkan adalah segmentasi otomatis.")
        processed,mask=result
        cols=st.columns(2)
        with cols[0]: show_image(processed,"Segmentasi area lesi")
        with cols[1]: show_image(mask,"Mask")
    elif isinstance(result,tuple):
        processed,mask=result
        cols=st.columns(2)
        with cols[0]: show_image(processed,"Processed")
        with cols[1]: show_image(mask,"Mask")
    else:
        cols=st.columns(2)
        with cols[0]: show_image(original,"Original")
        with cols[1]: show_image(result,method)


def page_training():
    st.title("🧠 Training CNN")
    st.write("Training demonstrasi CNN dengan dataset mini tiga kelas.")
    if dataset_error: st.error(dataset_error); return
    st.warning("⚠️ Dataset hanya 1 gambar per kelas. Augmentasi akan membuat variasi training, tetapi tidak menggantikan data independen. Hasil training/evaluasi tidak dapat dianggap sebagai performa generalisasi model.")
    with st.form("training_config"):
        epochs=st.slider("Epoch",1,30,5)
        batch_size=st.selectbox("Batch size",[2,4,8],index=1)
        image_size=st.selectbox("Image size",[96,128,160],index=1)
        start=st.form_submit_button("🚀 Mulai Training",type="primary")
    if start:
        progress=st.progress(0); status=st.empty()
        try:
            model,history,X_demo,y_demo,names=train_model(df,epochs,batch_size,image_size,progress_callback=progress)
            st.session_state["last_model"]=model
            st.session_state["demo_data"]=(X_demo,y_demo)
            st.session_state["class_names"]=names
            st.session_state["training_history"]=history
            progress.progress(100); status.success("Training demonstrasi selesai.")
            hist=history.history
            if "accuracy" in hist:
                st.subheader("Training Accuracy"); st.line_chart(pd.DataFrame({"accuracy":hist["accuracy"]}))
            if "loss" in hist:
                st.subheader("Training Loss"); st.line_chart(pd.DataFrame({"loss":hist["loss"]}))
            st.info("Training selesai pada data augmentasi dari sampel yang sama. Angka ini bukan validasi independen.")
        except Exception as e:
            st.error(f"Training gagal: {e}")


def page_evaluation():
    st.title("📊 Evaluation")
    if "last_model" not in st.session_state or "demo_data" not in st.session_state:
        st.info("Jalankan Training demonstrasi terlebih dahulu."); return
    st.warning("Evaluasi ini menggunakan sampel/variasi data yang sangat terbatas. Jangan gunakan metrik ini sebagai klaim performa penelitian atau diagnosis.")
    try:
        model=st.session_state["last_model"]
        X,y=st.session_state["demo_data"]
        names=st.session_state.get("class_names",["MEL","NV","BKL"])
        result=evaluate_model(model,X,y,names)
        st.metric("Demo Accuracy",f"{result['accuracy']*100:.2f}%")
        report=pd.DataFrame(result["report"]).T
        st.subheader("Classification Report"); st.dataframe(report,use_container_width=True)
        st.subheader("Confusion Matrix")
        cm=result["confusion_matrix"]
        fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm)
        ax.set_xticks(range(len(names)),names); ax.set_yticks(range(len(names)),names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(len(names)):
            for j in range(len(names)): ax.text(j,i,cm[i,j],ha="center",va="center")
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download report CSV",report.to_csv().encode("utf-8"),"demo_classification_report.csv","text/csv")
    except Exception as e: st.error(f"Evaluasi gagal: {e}")


if page=="Home": page_home()
elif page=="Dataset": page_dataset()
elif page in METHOD_DESCRIPTIONS: page_preprocess(page)
elif page=="Training (CNN)": page_training()
elif page=="Evaluation": page_evaluation()
