import tensorflow as tf
from tensorflow.keras import models, layers

# ---- Settings ----
DATA_DIR = "data"
IMG_SIZE = (160, 160)
BATCH_SIZE = 80
EPOCHS = 20

# ---- Load images from folders ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

# Optional but useful for faster loading
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ---- Build model ----
model = tf.keras.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---- Train model ----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# ---- Save model and names ----
model.save("plant_model.keras")
print("Training complete. Model saved to plant_model.keras")