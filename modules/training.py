
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 128
BATCH_SIZE = 16
CLASSES = ["MEL", "NV", "BKL"]


def prepare_data(df):

    if df is None or df.empty:
        raise ValueError("Dataset kosong.")

    images = []
    labels = []

    for _, row in df.iterrows():

        try:
            image = tf.keras.utils.load_img(
                row["path"],
                target_size=(IMG_SIZE, IMG_SIZE)
            )

            image = tf.keras.utils.img_to_array(image)

            images.append(image)
            labels.append(row["label"])

        except Exception:
            continue

    if len(images) == 0:
        raise ValueError("Tidak ada gambar yang berhasil dibaca.")

    X = np.array(images, dtype=np.float32) / 255.0

    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    y = encoder.transform(labels)

    return X, y, encoder


def build_model(num_classes=3):

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(
            32, 3,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(
            64, 3,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(
            128, 3,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(df, epochs=10, batch_size=BATCH_SIZE):

    X, y, encoder = prepare_data(df)

    if len(X) < 6:
        raise ValueError(
            "Jumlah gambar terlalu sedikit untuk training dan validation. "
            "Tambahkan dataset terlebih dahulu."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ])

    model = build_model(
        num_classes=len(encoder.classes_)
    )

    history = model.fit(
        augmentation(X_train, training=True),
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history, encoder
