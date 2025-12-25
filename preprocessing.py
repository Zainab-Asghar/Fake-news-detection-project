import argparse
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import emoji

# Common news sources to remove (prevents bias/memorization)
NEWS_SOURCES = [
    "reuters", "cnn", "fox", "foxnews", "ap", "associated press", "nbc", "abc",
    "bbc", "cbs", "msnbc", "the guardian", "washington post", "wsj", "nyt", "new york times"
]
AGENCY_REGEX = re.compile(r"\b(" + "|".join(re.escape(s) for s in NEWS_SOURCES) + r")\b", flags=re.IGNORECASE)

def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def remove_emojis(text: str) -> str:
    """Remove emojis and other non-text symbols."""
    return emoji.replace_emoji(text, replace=" ")

def clean_text(text: str) -> str:
    """Perform comprehensive cleaning on a given text."""
    if not isinstance(text, str):
        return ""

    text = clean_html(text)
    text = remove_emojis(text)
    text = text.lower()

    # Remove agency names
    text = AGENCY_REGEX.sub(" ", text)

    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)

    # Remove non-letter characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_text(text: str, max_words: int = 500) -> str:
    """Truncate text to limit extreme length bias."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def load_datasets(fake_path: str, true_path: str) -> pd.DataFrame:
    """Load and label Fake.csv and True.csv datasets."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    print(f"📥 Loaded {len(fake_df)} Fake and {len(true_df)} True samples (total: {len(df)})")
    return df


def clean_and_deduplicate(df: pd.DataFrame, max_words: int = 500) -> pd.DataFrame:
    """Clean, truncate, and deduplicate dataset."""
    before = len(df)

    # Drop missing or empty text
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].str.strip().astype(bool)]

    # Clean text and title
    df["title"] = df["title"].astype(str).apply(clean_text)
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Truncate
    df["text"] = df["text"].apply(lambda x: truncate_text(x, max_words))

    # Deduplicate
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    # Fill missing meta fields
    df["subject"] = df.get("subject", pd.Series(["unknown"] * len(df)))
    df["date"] = df.get("date", pd.Series(["unknown"] * len(df)))

    print(f"🧹 Cleaned from {before} → {len(df)} rows (removed {before - len(df)}).")
    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Balance dataset to avoid strong bias."""
    min_count = df["label"].value_counts().min()
    df_balanced = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
    )
    print(f"⚖️ Balanced dataset to {min_count} samples per class.")
    return df_balanced


def main():
    parser = argparse.ArgumentParser(description="Deep clean, deduplicate, and prepare fake news data.")
    parser.add_argument("--fake", type=str, required=True, help="Path to Fake.csv")
    parser.add_argument("--true", type=str, required=True, help="Path to True.csv")
    parser.add_argument("--output", type=str, required=True, help="Output cleaned CSV file path")
    parser.add_argument("--max-words", type=int, default=500, help="Max words per article")
    parser.add_argument("--balance", action="store_true", help="Enable class balancing")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = load_datasets(args.fake, args.true)
    df = clean_and_deduplicate(df, args.max_words)

    if args.balance:
        df = balance_classes(df)

    df.to_csv(args.output, index=False)
    print(f"✅ Saved cleaned dataset to: {args.output}")
    print(df['label'].value_counts())


if __name__ == "__main__":
    main()
