import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SIZE = (160, 160)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=True)
        tape.watch(conv_outputs)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        print("Error: Gradients are None. Check your model architecture.")
        return np.zeros((36, 36))
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, output_path="gradcam_result.jpg", alpha=0.4):
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * alpha + img
    superimposed = tf.keras.utils.array_to_img(superimposed)
    superimposed.save(output_path)
    print(f"Saved heatmap overlay to {output_path}")

if __name__ == "__main__":
    model = tf.keras.models.load_model("plant_model.keras")
    model.summary()
    last_conv_layer_name = "conv2d_2"
    img_path = "test_leaf.jpg"
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.cast(img_array, tf.float32)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay_heatmap(img_path, heatmap)
    