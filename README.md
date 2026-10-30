# Banana Freshness Detection Prototype

A simple Streamlit prototype that accepts a banana image and predicts one of three classes: **Fresh**, **Semi-Spoiled**, or **Spoiled**. It uses the MobileNetV2 transfer-learning model trained in the Google Colab workflow.

## 1. Export the trained model from Google Colab

After the training cell has completed, run this in a new Colab cell:

```python
from google.colab import files
files.download('/content/banana_freshness_model.keras')
files.download('/content/class_labels.json')
```

## 2. Put the model in the prototype folder

Copy `banana_freshness_model.keras` into this folder:

`Banana_Freshness_Detection_Prototype/models/banana_freshness_model.keras`

The `class_labels.json` file is already included. If downloaded from Colab, it can replace the included copy.

Alternatively, launch the app without a local model and upload the `.keras` file in the app's sidebar.

## 3. Install and launch (Windows)

Install Python 3.10 or 3.11, then open Command Prompt in this folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser, usually at `http://localhost:8501`.

## 4. Use the prototype

1. Upload `banana_freshness_model.keras` in the sidebar if it is not already in `models/`.
2. Upload a JPG, PNG or WEBP banana photograph.
3. Review the predicted class and all three model scores.

## Dataset and limitations

Training data: 30 labelled room-temperature page photographs (12 Fresh, 10 Semi-Spoiled, 8 Spoiled). Refrigerated images were excluded because their freshness labels were not verified. The reported held-out test result was 5/6 correct (83.3%) on a six-image test set. This is an extremely small evaluation and must not be presented as reliable real-world accuracy. The score displayed by the app is a model softmax output, not a calibrated probability. This prototype does not determine edibility, microbial contamination, or food safety.
