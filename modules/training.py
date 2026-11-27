from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers,models
from .dataset import CLASSES
MODEL_PATH=Path(__file__).resolve().parent.parent/"dermacnn_model.keras"

def train_model(df,epochs,batch_size,image_size,progress_callback=None):
    paths=[]; labels=[]
    for _,row in df.iterrows():
        paths.append(row.path); labels.append(CLASSES.index(row.label))
    if len(set(labels))<3: raise ValueError("Training memerlukan sampel dari semua 3 kelas.")
    X=[]; y=[]
    # Multiple random augmented views per original sample. This is demo training only.
    for path,label in zip(paths,labels):
        img=Image.open(path).convert("RGB").resize((image_size,image_size))
        arr=np.asarray(img,dtype=np.float32)/255.
        X.append(arr); y.append(label)
        for _ in range(8):
            aug=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,width_shift_range=.12,height_shift_range=.12,zoom_range=.15,horizontal_flip=True,brightness_range=(.8,1.2))
            batch=next(aug.flow(np.expand_dims(arr,0),batch_size=1,shuffle=False))
            X.append(batch[0]); y.append(label)
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.int32)
    ds=tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model=models.Sequential([layers.Input((image_size,image_size,3)),layers.Conv2D(16,3,padding="same",activation="relu"),layers.MaxPooling2D(),layers.Conv2D(32,3,padding="same",activation="relu"),layers.MaxPooling2D(),layers.Conv2D(64,3,padding="same",activation="relu"),layers.GlobalAveragePooling2D(),layers.Dropout(.25),layers.Dense(64,activation="relu"),layers.Dense(3,activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    callback=[]
    if progress_callback is not None:
        callback.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,logs:progress_callback.progress((epoch+1)/epochs)))
    history=model.fit(ds,epochs=int(epochs),verbose=0,callbacks=callback)
    model.save(MODEL_PATH)
    return model,history,(X,y),CLASSES
