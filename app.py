import gradio as gr
import tensorflow as tf
import numpy as np
import json

# Load the trained model
model = tf.keras.models.load_model("model/plant_identifier_efficientnetb0.keras")

# Load class indices
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the class_indices to map predicted index -> label
index_to_class = {v: k for k, v in class_indices.items()}

def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array[np.newaxis, ...]

    # Predict
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label = index_to_class[predicted_index]

    return f"This looks like a {label} ({confidence:.2%} confidence)."

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Plant Identifier",
    description="Upload an image of a plant, and this AI will tell you what type it is.",
    theme="default",
)

demo.launch()
