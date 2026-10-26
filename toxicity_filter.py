"""Cyberbullying & Toxicity Filter - NLP pipeline.

Pipeline: input -> cleaning -> tokenization -> stop-word removal -> stemming
-> TF-IDF features -> model prediction -> severity score -> explainable AI
-> final moderation output.

Usage:
    python toxicity_filter.py train                      # train on data/sample_comments.csv
    python toxicity_filter.py train --jigsaw train.csv   # train on the Jigsaw dataset
    python toxicity_filter.py predict "your comment"
"""
import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
SAMPLE_DATA = BASE / "data" / "sample_comments.csv"

LABELS = ["safe", "mildly rude", "toxic"]

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "and",
    "or", "but", "of", "to", "in", "on", "at", "for", "with", "this", "that",
    "it", "its", "as", "by", "from", "so", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "just", "then", "there", "here",
}


# Abusive-word lexicon (hybrid approach: lexicon + ML model). Edit freely.
MILD_WORDS = {"stupid", "dumb", "idiot", "loser", "moron", "pathetic", "annoying", "lame",
              "garbage", "trash", "clown", "fool", "useless", "clueless", "crappy", "lazy",
              "ridiculous", "pointless", "childish", "irritating", "awful", "embarrassing"}
STRONG_WORDS = {"worthless", "disgusting", "freak", "filthy", "vile", "creep", "ugly",
                "scum", "waste", "coward", "failure"}
THREAT_PHRASES = ["kill yourself", "i will hurt", "i will find you", "i know where you live",
                  "go die", "drop dead", "beat you up", "make you pay"]
NEVER_HIGHLIGHT = {"i", "you", "your", "you're", "me", "my", "we", "he", "she", "they", "such", "nobody"}


# ---------- Steps 2-5: preprocessing ----------
def clean_text(text: str) -> str:
    """Lowercase; remove URLs, emojis, punctuation, special characters, extra spaces."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return text.split()


def remove_stop_words(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS]


def stem(word: str) -> str:
    """Light rule-based stemmer (no external downloads needed)."""
    if word.endswith("ss"):
        return word
    for suffix in ("ing", "edly", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess(text: str) -> str:
    tokens = remove_stop_words(tokenize(clean_text(text)))
    return " ".join(stem(t) for t in tokens)


# ---------- Training ----------
def load_jigsaw(path: str) -> pd.DataFrame:
    """Map Jigsaw Toxic Comment columns to 3 levels."""
    df = pd.read_csv(path)
    severe = df[["severe_toxic", "threat", "identity_hate"]].max(axis=1) == 1
    mild = df[["toxic", "obscene", "insult"]].max(axis=1) == 1
    label = pd.Series("safe", index=df.index)
    label[mild] = "mildly rude"
    label[severe] = "toxic"
    return pd.DataFrame({"text": df["comment_text"], "label": label})


def train(data: pd.DataFrame) -> dict:
    data = data.dropna()
    y = data["label"].map({l: i for i, l in enumerate(LABELS)})
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], y, test_size=0.25, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(preprocessor=preprocess, token_pattern=r"[a-z]+")
    model = LogisticRegression(C=10, max_iter=1000, class_weight="balanced")
    model.fit(vectorizer.fit_transform(X_train), y_train)
    preds = model.predict(vectorizer.transform(X_test))
    print(classification_report(y_test, preds, target_names=LABELS, zero_division=0))
    bundle = {"vectorizer": vectorizer, "model": model}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        train(pd.read_csv(SAMPLE_DATA))
    return joblib.load(MODEL_PATH)


# ---------- Steps 6-10: prediction, severity, XAI, output ----------
def suggest_action(score: float) -> str:
    if score < 25:
        return "Allow - no action needed"
    if score < 60:
        return "Flag for moderator review and send a gentle warning to the user"
    return "Remove comment and escalate to a moderator"


def analyze(text: str, bundle: dict = None) -> dict:
    bundle = bundle or load_bundle()
    vec, model = bundle["vectorizer"], bundle["model"]
    X = vec.transform([text])
    probs = model.predict_proba(X)[0]
    pred = int(probs.argmax())

    # Severity: expected level (safe=0, mildly rude=50, toxic=100)
    severity = round(float(50 * probs[1] + 100 * probs[2]), 1)

    # Lexicon check (works alongside the ML model)
    tokens = set(clean_text(text).split())
    mild_hits = tokens & MILD_WORDS
    strong_hits = tokens & STRONG_WORDS
    threat = any(p in clean_text(text) for p in THREAT_PHRASES)
    if threat or len(strong_hits) >= 2:
        floor = 2
    elif mild_hits or strong_hits:
        floor = 1
    else:
        floor = 0
    pred = max(pred, floor)
    severity = max(severity, {0: 0, 1: 30, 2: 70}[floor])

    # XAI: highlight lexicon words plus words with a strong model contribution
    flagged = set(mild_hits | strong_hits)
    if pred != 0 and not flagged:
        vocab = vec.vocabulary_
        idx_to_stem = {i: st for st, i in vocab.items()}
        row = X.tocoo()
        contrib = {j: model.coef_[min(pred, 2)][j] * v for j, v in zip(row.col, row.data)}
        positive = {j: c for j, c in contrib.items() if c > 0}
        if positive:
            cutoff = 0.5 * max(positive.values())
            flagged |= {idx_to_stem[j] for j, c in positive.items() if c >= cutoff}

    words = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    highlighted = []
    for w in words:
        lw = w.lower()
        is_word = bool(re.match(r"[A-Za-z']+$", w))
        hit = is_word and lw not in NEVER_HIGHLIGHT and (lw in flagged or stem(lw) in flagged)
        highlighted.append((w, hit))
    keywords = sorted({w.lower() for w, hit in highlighted if hit})
    if threat and not keywords:
        keywords = ["threatening phrase"]

    return {
        "category": LABELS[pred],
        "probabilities": {l: round(float(p), 3) for l, p in zip(LABELS, probs)},
        "toxicity_probability": round(float(probs[1] + probs[2]), 3),
        "severity_score": severity,
        "highlighted": highlighted,
        "keywords": keywords,
        "explanation": explain(LABELS[pred], keywords, severity),
        "action": suggest_action(severity),
    }


def explain(category: str, keywords: list, severity: float) -> str:
    """Template-based explanation. To use Generative AI instead, send the
    category, keywords and severity to an LLM API and return its response here."""
    if category == "safe":
        return "No abusive language or aggressive tone was detected, so this comment follows community guidelines."
    quoted = ", ".join(f'"{k}"' for k in keywords) or "the overall tone"
    if category == "mildly rude":
        return (f"The comment was classified as mildly rude (severity {severity}/100) because it contains "
                f"insulting or dismissive language: {quoted}. It is disrespectful but not a direct threat.")
    return (f"The comment was classified as toxic (severity {severity}/100) because it contains abusive, "
            f"threatening or harassing language: {quoted}. This violates community guidelines on harassment.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--jigsaw", help="path to Jigsaw train.csv")
    p = sub.add_parser("predict")
    p.add_argument("text")
    args = ap.parse_args()

    # Use the module by its real name so saved models can be loaded by app.py
    import toxicity_filter as tf

    if args.cmd == "train":
        tf.train(tf.load_jigsaw(args.jigsaw) if args.jigsaw else pd.read_csv(SAMPLE_DATA))
    else:
        r = tf.analyze(args.text)
        for k in ("category", "toxicity_probability", "severity_score", "keywords", "explanation", "action"):
            print(f"{k}: {r[k]}")
