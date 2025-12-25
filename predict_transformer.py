# scripts/predict_transformer.py
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def load_pipeline(model_dir: str):
    tokenizer= AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Force config mapping just in case
    model.config.id2label = getattr(model.config, "id2label", {0:"FAKE",1:"REAL"})
    model.config.label2id = getattr(model.config, "label2id", {"FAKE":0,"REAL":1})
    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, return_all_scores=False, truncation=True)
    return pipe

def main():
    parser = argparse.ArgumentParser(description="Predict with a fine-tuned BERT fake-news model")
    parser.add_argument("--model-dir", "-m", default="models/distilbert-fake-news", help="Path to saved model dir")
    parser.add_argument("--text", "-t", help='Single text to classify (e.g., -t "This is a news article")')
    parser.add_argument("--input-csv", help="CSV with a text column")
    parser.add_argument("--text-column", default="text", help="Name of text column in CSV")
    parser.add_argument("--output-csv", help="Where to save CSV with predictions")
    args = parser.parse_args()

    pipe = load_pipeline(args.model_dir)

    if args.text:
        pred = pipe(args.text)[0]  # {'label': 'REAL'/'FAKE', 'score': ...}
        print(f"Prediction: {pred['label']}  |  confidence: {pred['score']:.3f}")
        return

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.text_column not in df.columns:
            raise ValueError(f"CSV must contain '{args.text_column}' column")
        texts = df[args.text_column].fillna("").astype(str).tolist()
        results = pipe(texts)
        df["pred_label"] = [r["label"] for r in results]
        df["pred_conf"] = [r["score"] for r in results]
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved predictions to {args.output_csv}")
        else:
            print(df[["pred_label", "pred_conf"]].head())
        return

    raise SystemExit("Provide either --text or --input-csv")

if __name__ == "__main__":
    main()
