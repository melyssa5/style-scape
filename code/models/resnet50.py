from tkinter import Image
import cv2
import tensorflow as tf
import numpy as np
import os
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
     
class ResNet50(tf.keras.Model):
       def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()

        # Load the ResNet50 base model
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet' if pretrained else None,  # ← Toggle pretrained vs. scratch
            input_shape=(224, 224, 3),
            pooling='avg'
        )

        # Freeze the base model if pretrained
        self.base_model.trainable = not pretrained  # If pretrained, freeze; else train all

        # Classification head for 15-scene classification
        self.head = tf.keras.Sequential([
            Dense(256, activation='relu', name='dense1'),
            Dropout(0.3, name='dropout1'),
            Dense(128, activation='relu', name='dense2'),
            Dropout(0.3, name='dropout2'),
            Dense(units=hp.num_classes, activation='softmax', name='output_layer')
        ])

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

       def call(self, x):
              x = tf.keras.applications.resnet.preprocess_input(x * 255.0)  # Match ResNet preprocessing
              x = self.base_model(x)
              x = self.head(x)
              return x

       @staticmethod
       def loss_fn(labels, predictions):
              cce = tf.keras.losses.SparseCategoricalCrossentropy()
              return cce(labels, predictions)
       


data_dir = "../../data"  
weight_path = "resnet50.e013-acc0.6412.weights.h5" 
img_size = (224, 224)
batch_size = 32
num_classes = 15

# Rebuilding model from saved weights
def build_model():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='dense')(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

model = build_model()

# 👇 Try loading with skip_mismatch=True
model.load_weights(weight_path, skip_mismatch=True)


model.summary()

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "test"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
stylized_test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_dir, "stylized"),
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
class_names = test_ds.class_names


#Lime Function 
def run_lime_interpreter(model, image_np, class_names, true_label=None, pred_label=None,
                         save_dir="lime_outputs", filename_prefix=None):
    """
    Run LIME explanation and save to a directory.
    """

    os.makedirs(save_dir, exist_ok=True)

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(imgs):
        imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
        imgs = tf.keras.applications.resnet50.preprocess_input(imgs)
        return model.predict(imgs)

    if pred_label is None:
        img_batch = np.expand_dims(image_np, axis=0)
        pred_label = np.argmax(predict_fn(img_batch), axis=1)[0]

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        labels=[pred_label],
        top_labels=1,
        num_samples=1000,
        hide_color=0
    )

    temp, mask = explanation.get_image_and_mask(
        label=pred_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # === Build filename ===
    true_str = f"{class_names[true_label]}" if true_label is not None else "unknown"
    pred_str = f"{class_names[pred_label]}"
    name = f"{filename_prefix}_" if filename_prefix else ""
    name += f"pred_{pred_str}_true_{true_str}.png"
    save_path = os.path.join(save_dir, name)

    # === Save figure ===
    plt.figure(figsize=(6, 6))
    plt.title(f"LIME | Pred: {pred_str} | True: {true_str}")
    # Ensure float image is in [0, 1] for correct display
    temp = np.array(temp) / 255.0 if temp.max() > 1 else temp
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def run_gradcam_interpreter(model, image_np, class_names, true_label=None, pred_label=None,
                            layer_name="conv5_block3_out", save_dir="gradcam_outputs", filename_prefix=None):
    """
    Runs Grad-CAM and saves both the original image and the heatmap overlay to file.
    """
    os.makedirs(save_dir, exist_ok=True)
    original_dir = os.path.join(save_dir, "originals")
    os.makedirs(original_dir, exist_ok=True)

    #  Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    #  Forward and compute gradients
    img_input = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(image_np, axis=0))
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        if pred_label is None:
            pred_label = tf.argmax(predictions[0])
        loss = predictions[:, pred_label]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize and overlay
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_rgb = np.uint8(255 * plt.cm.jet(heatmap)[:, :, :3])
    image_rgb = image_np.astype(np.uint8)
    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # Save both original and Grad-CAM
    true_str = class_names[true_label] if true_label is not None else "unknown"
    pred_str = class_names[pred_label]
    base_name = f"{filename_prefix or 'sample'}_pred_{pred_str}_true_{true_str}.png"

    gradcam_path = os.path.join(save_dir, base_name)
    original_path = os.path.join(original_dir, base_name)

    Image.fromarray(image_rgb).save(original_path)
    plt.imsave(gradcam_path, overlay)
    print(f"Saved Grad-CAM: {gradcam_path}")
    print(f"Saved Original Image: {original_path}")


# ====== MAIN ======
if __name__ == "__main__":
    # print(f"\n✅ Natural Test Set Evaluation Accuracy")
    # evaluate(model, test_ds)

    # print(f"\n✅ Stylized Test Set Evaluation Accuracy")
    # evaluate(model, stylized_test_ds)
# Load stylized data with file paths
    stylized_path = "../../data/stylized"
    stylized_ds, class_names = load_stylized_dataset_with_paths(stylized_path)

    # Pick explanation samples!




