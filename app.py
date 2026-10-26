import io

import pandas as pd
from flask import Flask, jsonify, render_template, request

from toxicity_filter import analyze, load_bundle

app = Flask(__name__)
BUNDLE = load_bundle()

MAX_CHARS = 5000
MAX_ROWS = 5000



@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Enter a comment to check."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Comments can be up to {MAX_CHARS:,} characters."), 400
    return jsonify(analyze(text, BUNDLE))


@app.post("/api/batch")
def api_batch():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception:
        return jsonify(error="That file could not be read as a CSV."), 400

    columns = {c.lower().strip(): c for c in df.columns}
    if "text" not in columns:
        return jsonify(error="The file needs a column named text."), 400
    df = df.dropna(subset=[columns["text"]]).head(MAX_ROWS)

    rows = []
    for text in df[columns["text"]].astype(str):
        r = analyze(text, BUNDLE)
        rows.append({
            "text": text,
            "category": r["category"],
            "severity_score": r["severity_score"],
            "keywords": r["keywords"],
            "action": r["action"],
        })
    return jsonify(rows=rows)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
