from pathlib import Path
import json
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Banana Freshness Detection Prototype", page_icon="🍌", layout="wide")

CLASSES = ["Fresh", "Semi-Spoiled", "Spoiled"]
IMG_SIZE = (160, 160)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "banana_freshness_model.keras"
LABELS_PATH = Path(__file__).resolve().parent / "models" / "class_labels.json"

st.title("AI-Based Banana Freshness Detection")
st.caption("Prototype demonstration | MobileNetV2 transfer learning")
st.write(
    "Upload a banana photograph to obtain a model prediction for Fresh, Semi-Spoiled, or Spoiled. "
    "This is an academic proof of concept trained on a very small dataset; predictions are not a food-safety assessment."
)

with st.sidebar:
    st.header("Model setup")
    st.write("The trained model file is required. Upload it here, or place it in the `models` folder.")
    model_upload = st.file_uploader("Upload trained model (.keras)", type=["keras"])
    labels_upload = st.file_uploader("Optional class labels (.json)", type=["json"])
    st.divider()
    st.subheader("Dataset used")
    st.write("30 labelled room-temperature images")
    st.write("Fresh: 12 · Semi-Spoiled: 10 · Spoiled: 8")
    st.caption("Refrigerated images were excluded because verified class labels were not provided.")

@st.cache_resource(show_spinner="Loading model…")
def load_model_from_path(path_str):
    return tf.keras.models.load_model(path_str, compile=False)

@st.cache_resource(show_spinner="Loading uploaded model…")
def load_model_from_bytes(model_bytes):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(model_bytes)
        temp_path = tmp.name
    try:
        model = tf.keras.models.load_model(temp_path, compile=False)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass
    return model

model = None
load_error = None
try:
    if model_upload is not None:
        model = load_model_from_bytes(model_upload.getvalue())
    elif MODEL_PATH.exists():
        model = load_model_from_path(str(MODEL_PATH))
except Exception as exc:
    load_error = str(exc)

if labels_upload is not None:
    try:
        label_map = json.loads(labels_upload.getvalue().decode("utf-8"))
        classes = [label_map[str(i)] for i in range(len(label_map))]
    except Exception:
        classes = CLASSES
else:
    classes = CLASSES

if model is None:
    st.warning("No trained model is loaded. Upload `banana_freshness_model.keras` using the sidebar, or add it to `models/` and restart the app.")
    if load_error:
        st.error(f"Model loading error: {load_error}")
    st.stop()

uploaded_image = st.file_uploader("Choose a banana image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Input image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Prediction")
            resized = image.resize(IMG_SIZE)
            arr = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(arr, axis=0)
            # Matches the training pipeline: MobileNetV2 preprocessing inside the model.
            predictions = model.predict(batch, verbose=0)[0]
            if len(predictions) != len(classes):
                st.error(f"Model returned {len(predictions)} outputs but {len(classes)} class labels were supplied.")
                st.stop()
            best_idx = int(np.argmax(predictions))
            best_class = classes[best_idx]
            confidence = float(predictions[best_idx])
            st.metric("Predicted category", best_class)
            st.metric("Model score", f"{confidence * 100:.1f}%")
            st.caption("The score is the model's softmax output, not a calibrated probability of actual freshness.")
            st.write("Class scores")
            for name, score in zip(classes, predictions):
                st.write(f"**{name}** — {score * 100:.1f}%")
                st.progress(float(np.clip(score, 0, 1)))

            if confidence < 0.60:
                st.warning("Low model confidence. Treat this result as uncertain and verify manually.")
            elif best_class == "Spoiled":
                st.warning("The model predicts Spoiled. This is not a laboratory test and does not determine food safety.")
            else:
                st.info("Prototype result only. Visual appearance alone cannot establish edibility or microbiological safety.")

        with st.expander("Prototype limitations and intended use"):
            st.write(
                "The model was trained on only 30 labelled page photographs (12 Fresh, 10 Semi-Spoiled, 8 Spoiled). "
                "The held-out test set contains six images, so one error changes test accuracy by 16.7 percentage points. "
                "Related photos of the same fruit may overlap between data splits. Lighting, camera, background, variety, "
                "ripeness and storage conditions can affect predictions. Do not use this prototype to decide whether food is safe to eat."
            )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload an image to run the prototype prediction.")

st.divider()
st.caption("Project prototype — not validated for commercial, clinical, or food-safety decisions.")
