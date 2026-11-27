from pathlib import Path
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
IMAGE_DIR=ROOT/"dataset"/"images"
CLASSES=["MEL","NV","BKL"]

def load_dataset_index():
    rows=[]
    for label in CLASSES:
        folder=IMAGE_DIR/label
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
                    try:
                        with Image.open(p) as im: im.verify()
                        rows.append({"image_id":p.stem,"label":label,"path":str(p)})
                    except Exception: pass
    df=pd.DataFrame(rows,columns=["image_id","label","path"])
    if df.empty: return df,"Folder gambar kosong. Pastikan gambar tersedia di dataset/images/MEL, NV, dan BKL."
    missing=[c for c in CLASSES if c not in set(df.label)]
    if missing: return df,f"Kelas belum memiliki gambar: {', '.join(missing)}"
    return df,None

def class_summary(df):
    return {c:int((df.label==c).sum()) for c in CLASSES}

def sample_image(df,label):
    subset=df[df.label==label]
    if subset.empty: return None,None
    row=subset.iloc[0]
    try: return Image.open(row.path).convert("RGB"),row.image_id
    except Exception: return None,row.image_id
