# mnist_app_clean.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = r"F:\Projects\MINST-CNN\mnist_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("Handwritten Digit Recognition")

canvas_result = st_canvas(
    stroke_width=27,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_canvas_pil(img_data, invert=False, thresh=30):
    """
    img_data: HxWx4 RGBA numpy array from streamlit canvas
    returns: (1,28,28,1) float32 normalized to 0-1
    """
    # 1) RGBA -> grayscale
    gray = np.mean(img_data[:, :, :3], axis=2).astype(np.uint8)

    # 2) optionally invert (we're using invert=False as requested)
    if invert:
        gray = 255 - gray

    # 3) binarize using threshold to find bounding box
    bw = (gray > thresh).astype(np.uint8) * 255
    coords = np.argwhere(bw)
    if coords.size == 0:
        img28 = np.zeros((28,28), dtype=np.uint8)
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        digit = gray[y0:y1+1, x0:x1+1]

        # 4) resize keeping aspect ratio to fit into 20x20 box
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = max(1, int(round(w * 20.0 / h)))
        else:
            new_w = 20
            new_h = max(1, int(round(h * 20.0 / w)))

        digit_img = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 5) center in 28x28 canvas
        result = Image.new('L', (28, 28), 0)
        x_pad = (28 - new_w) // 2
        y_pad = (28 - new_h) // 2
        result.paste(digit_img, (x_pad, y_pad))
        img28 = np.array(result)

    # 6) normalize to 0-1 (model trained on normalized data)
    img28 = img28.astype("float32") / 255.0
    return img28.reshape(1, 28, 28, 1)

# Prediction
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = preprocess_canvas_pil(canvas_result.image_data, invert=False, thresh=30)
        pred = np.argmax(model.predict(img)[0])

        st.image(img.reshape(28,28), caption="Preprocessed 28×28 Input", width=150)
        st.write(f"### Predicted Label: {pred}")
    else:
        st.warning("Draw a digit first.")
