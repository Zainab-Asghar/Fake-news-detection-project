from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Path to your saved model folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "distilbert-fake-news")

# Load tokenizer & model once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Build label mapping robustly
id2label = getattr(model.config, "id2label", None) or {}

def normalize_label(name, idx):
    if isinstance(name, str) and name.upper() in ("FAKE", "REAL"):
        return name.upper()
    if isinstance(name, str) and name.upper().startswith("LABEL_"):
        return "FAKE" if idx == 0 else "REAL"
    return str(name)

# Corrected: iterate id2label.items() directly (no double .items())
if id2label:
    id2label_normalized = {int(k): normalize_label(v, int(k)) for k, v in id2label.items()}
else:
    id2label_normalized = {0: "FAKE", 1: "REAL"}


app = Flask(__name__, template_folder="templates", static_folder="static")

def predict_text(text, max_length=512):
    """Return (label_str, confidence_float, probs_list)"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]  # e.g. [0.01, 0.99]
    idx = int(probs.argmax())
    label = id2label_normalized.get(idx, f"LABEL_{idx}")
    confidence = float(probs[idx])
    return label, confidence, probs.tolist()

@app.route("/", methods=["GET"])
def home():
    # initial page load - no prediction yet
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text", "")
    if not text or not text.strip():
        return render_template("index.html")

    # Run model prediction
    label, conf, probs = predict_text(text)
    confidence_percent = round(conf * 100, 2)

    # Disclaimer rule (ethical AI practice)
    show_disclaimer = confidence_percent < 95

    # Optional terminal log (no persistent storage)
    print(f"Predicted: {label} | confidence: {conf:.6f} | probs: {probs}")

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence_percent,
        show_disclaimer=show_disclaimer
    )


if __name__ == "__main__":
    # development server. In production use gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
