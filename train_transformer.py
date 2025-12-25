# ml_scripts/train_transformer.py
"""
Robust trainer for FakeNewsModel (version-safe TrainingArguments).
- Merges title+text (if present).
- Tokenizes safely.
- Uses max_length (default 512).
- Sets reproducible seeds.
- Applies early stopping.
- Builds TrainingArguments kwargs, prunes to installed HF version,
  ensures consistency for load_best_model_at_end.
"""

import argparse
import inspect
import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)


# ---------- helpers ----------
def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def combine_title_text_batch(batch):
    # batch is a dict of lists
    titles = batch.get("title", None)
    texts = batch.get("text", None)
    if titles is not None and texts is not None:
        return [(t if t else "") + " " + (b if b else "") for t, b in zip(titles, texts)]
    elif texts is not None:
        return [t if t else "" for t in texts]
    else:
        raise ValueError("Dataset must contain 'text' column or both 'title' and 'text'.")


def preprocess_function(examples, tokenizer, max_length):
    # examples is a dict of lists
    combined = combine_title_text_batch(examples)
    combined = ["" if x is None else str(x) for x in combined]
    return tokenizer(combined, truncation=True, padding=False, max_length=max_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def build_pruned_kwargs(desired_kwargs):
    """
    Return (pruned_kwargs, unsupported_keys)
    where pruned_kwargs only contains keys supported by the installed TrainingArguments.
    """
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    allowed = set(ta_sig.keys())
    pruned = {k: v for k, v in desired_kwargs.items() if k in allowed and v is not None}
    unsupported = [k for k in desired_kwargs.keys() if k not in allowed]
    return pruned, unsupported


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Version-safe Transformer trainer")
    parser.add_argument("--train-file", required=True, help="Path to train CSV (must have 'text' or 'title'+'text' and 'label')")
    parser.add_argument("--val-file", required=True, help="Path to val CSV")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="./models/distilbert-fake-news")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("🚀 Starting trainer (version-safe)")

    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load CSVs
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)

    # basic checks & clean
    if "label" not in train_df.columns or "label" not in val_df.columns:
        raise ValueError("Both train and val CSV must contain a 'label' column.")

    # ensure text/title exist
    if not (("text" in train_df.columns) or (("title" in train_df.columns) and ("text" in train_df.columns))):
        raise ValueError("Train CSV must contain 'text' or both 'title' and 'text' columns.")
    if not (("text" in val_df.columns) or (("title" in val_df.columns) and ("text" in val_df.columns))):
        raise ValueError("Val CSV must contain 'text' or both 'title' and 'text' columns.")

    # coerce types and fill missing
    for col in ["title", "text"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("").astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna("").astype(str)
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    # model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    print("🔤 Tokenizing train dataset...")
    train_ds = train_ds.map(lambda ex: preprocess_function(ex, tokenizer, args.max_length), batched=True)
    print("🔤 Tokenizing val dataset...")
    val_ds = val_ds.map(lambda ex: preprocess_function(ex, tokenizer, args.max_length), batched=True)

    # rename label -> labels
    if "label" in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
    if "label" in val_ds.column_names:
        val_ds = val_ds.rename_column("label", "labels")

    # set torch format
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # desired kwargs (modern)
    desired = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        # F1 increases => greater is better
        "greater_is_better": True,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "logging_steps": 50,
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
    }

    pruned, unsupported = build_pruned_kwargs(desired)
    if unsupported:
        print("ℹ️ The installed transformers TrainingArguments does not support these kwargs and they will be ignored:", unsupported)

    # If load_best_model_at_end plan exists, ensure eval/save strategies are present and matching
    if pruned.get("load_best_model_at_end", False):
        # The training args must include evaluation and save strategy and they must match.
        eval_key = None
        save_key = None
        # Check which keys are present in signature instead of relying on names
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params and "save_strategy" in ta_params:
            eval_key = "evaluation_strategy"
            save_key = "save_strategy"
        elif "eval_strategy" in ta_params and "save_steps" in ta_params:
            # very old API: can't match strategies easily; fallback to disabling load_best
            eval_key = None
            save_key = None

        if eval_key is None or save_key is None:
            # cannot guarantee matching strategies on this HF version -> disable best model loading
            print("⚠️ Cannot set matching eval/save strategy on this transformers version. Disabling load_best_model_at_end.")
            pruned.pop("load_best_model_at_end", None)
            pruned.pop("metric_for_best_model", None)
            pruned.pop("greater_is_better", None)
        else:
            # ensure the values exist and match; if not, force them to match
            ev = pruned.get(eval_key)
            sv = pruned.get(save_key)
            if ev is None and sv is None:
                # nothing to do; evaluation may be disabled -> disable load_best
                print("⚠️ evaluation/save strategy missing -> disabling load_best_model_at_end")
                pruned.pop("load_best_model_at_end", None)
                pruned.pop("metric_for_best_model", None)
                pruned.pop("greater_is_better", None)
            elif ev is None and sv is not None:
                # set evaluation strategy to save strategy
                pruned[eval_key] = sv
                print(f"ℹ️ Setting {eval_key} to match {save_key}: {sv}")
            elif ev is not None and sv is None:
                pruned[save_key] = ev
                print(f"ℹ️ Setting {save_key} to match {eval_key}: {ev}")
            elif ev != sv:
                # force them to match (preferring 'epoch' if possible)
                preferred = "epoch" if "epoch" in (ev, sv, "") else ev
                pruned[eval_key] = preferred
                pruned[save_key] = preferred
                print(f"ℹ️ Forcing evaluation/save strategy match to: {preferred}")

    # finally create TrainingArguments safely
    try:
        training_args = TrainingArguments(**pruned)
    except TypeError as e:
        # Last-resort prune: remove keys not in signature just in case
        print("⚠️ TrainingArguments init raised TypeError; attempting a more conservative prune:", e)
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        allowed = set(ta_params.keys())
        conservative = {k: v for k, v in pruned.items() if k in allowed}
        training_args = TrainingArguments(**conservative)
    except ValueError as e:
        # If ValueError (like mismatched strategies), try to disable load_best_model_at_end and retry
        print("⚠️ TrainingArguments init raised ValueError; retrying without load_best_model_at_end:", e)
        pruned.pop("load_best_model_at_end", None)
        pruned.pop("metric_for_best_model", None)
        pruned.pop("greater_is_better", None)
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        allowed = set(ta_params.keys())
        conservative = {k: v for k, v in pruned.items() if k in allowed}
        training_args = TrainingArguments(**conservative)

    print("✅ TrainingArguments keys used:", list(training_args.to_dict().keys()))

    # build trainer
    # Decide whether to add EarlyStoppingCallback: only add if metric_for_best_model exists and load_best_model_at_end is True.
    trainer_callbacks = []
    metric_for_best = getattr(training_args, "metric_for_best_model", None)
    load_best_flag = getattr(training_args, "load_best_model_at_end", False)

    if load_best_flag and metric_for_best:
        trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        print(f"ℹ️ EarlyStoppingCallback enabled (monitoring '{metric_for_best}').")
    else:
        # Avoid adding the callback if the metric_for_best_model isn't set or load_best_model_at_end was disabled/pruned.
        print("⚠️ EarlyStoppingCallback NOT enabled because 'metric_for_best_model' or 'load_best_model_at_end' is not set on TrainingArguments.")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=trainer_callbacks,
    )

    print("🚀 Starting training...")
    trainer.train()

    # save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("🎉 Training complete. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
