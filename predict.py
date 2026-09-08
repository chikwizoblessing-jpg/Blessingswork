import tensorflow as tf
import numpy as np

IMG_SIZE = (160, 160)

# Load the trained model
model = tf.keras.models.load_model("plant_model.keras")

# Load the class names we saved earlier
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---- Change this to the path of a leaf photo you want to test ----
img_path = "test_leaf.jpg"
img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # model expects a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
