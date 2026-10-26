# Cyberbullying & Toxicity Filter

NLP mini project: classifies comments as safe, mildly rude or toxic, gives a severity score (0-100),
marks the abusive words, explains the decision and suggests a moderation action.

## Run it

```bash
pip install -r requirements.txt
python toxicity_filter.py train      # trains on data/sample_comments.csv, saves model.joblib
python app.py                        # then open http://localhost:5000
```

Command-line check without the web page:

```bash
python toxicity_filter.py predict "You are a pathetic loser"
```

## Project layout

- `toxicity_filter.py` - the NLP pipeline, model, severity score and explanation
- `app.py` - Flask server with `/api/analyze` and `/api/batch`
- `templates/index.html`, `static/style.css`, `static/app.js` - the web interface
- `data/sample_comments.csv` - 72 labelled demo comments

## Pipeline (matches the write-up)

1. Input collection (text box or CSV file)
2. Text cleaning (lowercase, remove URLs/emojis/punctuation)
3. Tokenization
4. Stop-word removal
5. Stemming
6. Feature extraction: TF-IDF
7. Model prediction: Logistic Regression (3 classes), combined with an abusive-word list
8. Severity score = 50 x P(mildly rude) + 100 x P(toxic), raised to a minimum when the word list matches
9. Explainable AI: abusive words are marked in the comment; if the list has no match, words are chosen by coefficient x TF-IDF weight
10. Final output: category, probability, severity, marked words, explanation, suggested action

## Better accuracy

The bundled dataset has only 72 comments, so it is a demo. For real results, download the
Jigsaw Toxic Comment dataset from Kaggle and run:

```bash
python toxicity_filter.py train --jigsaw train.csv
```

## Generative AI part

`explain()` in `toxicity_filter.py` is template-based. To use a generative model, send the category,
keywords and severity to an LLM API there and return its text.
