# AI Model Explanation

This document explains the AI part of this project for someone who has never seen the code before.

The short version is:

- The project analyzes a journal entry in two ways.
- It uses rule-based text matching to find explicit cognitive distortions.
- It uses a trained PyTorch model to estimate whether the text sounds distorted and what emotion is most likely present.
- The Flask API combines both results into one response.

## 1. What the model is trying to do

The goal of the model is to read a journal entry and answer questions like:

- Does this text contain cognitive distortions?
- Which emotion is most likely expressed?
- How concerning does the entry seem overall?

This is not a large language model or a chatbot. It is a lightweight classification pipeline built from:

- a text vectorizer,
- a binary neural network for distortion detection,
- a multi-class neural network for emotion classification,
- and some rule-based pattern matching for explainable distortion examples.

## 2. Where the AI code lives

The main implementation is in [model.py](model.py).

The API path that uses it is:

- [routes/analyze.py](routes/analyze.py)
- [services/analysis.py](services/analysis.py)
- [app.py](app.py)

The most important function for the rest of the application is `get_analyzer()` in [model.py](model.py). That function returns a ready-to-use analyzer object.

## 3. High-level flow

When a user sends text to the API, the flow looks like this:

1. The Flask route receives the text.
2. The service layer runs rule-based distortion matching.
3. The service layer asks the ML analyzer for a deeper prediction.
4. The analyzer converts the text into numeric features.
5. The distortion model predicts a score between 0 and 1.
6. The emotion model predicts the most likely emotion class.
7. The service layer merges everything into one JSON response.

If the ML model cannot be loaded, the service still works with the rule-based fallback.

## 4. The pieces inside `model.py`

### `DistortionClassifier`

This is a small feed-forward neural network used for binary classification.

Its job is to decide whether the journal entry looks distorted or not.

The network structure is:

- input layer based on the vectorized text size,
- hidden layer with 256 units,
- hidden layer with 64 units,
- hidden layer with 32 units,
- final output layer with 1 unit,
- sigmoid activation at the end.

Because the last layer uses sigmoid, the output is a number between 0 and 1. A value above 0.5 is treated as distorted.

### `EmotionClassifier`

This is another feed-forward neural network, but it predicts one emotion from several possible emotion labels.

Its job is to answer: “What emotion is the text most likely expressing?”

The network structure is:

- input layer based on the same vectorized text,
- hidden layer with 256 units,
- hidden layer with 64 units,
- final layer with one output per emotion class.

The model returns raw scores, also called logits. In the analysis step, those logits are converted into probabilities with softmax.

### `CustomJournalAnalyzer`

This is the main analyzer object used by the rest of the app.

It combines:

- the distortion model,
- the emotion model,
- the TF-IDF vectorizer,
- and the emotion label encoder.

It also includes a set of regex patterns that look for specific distortion types such as:

- all-or-nothing thinking,
- overgeneralization,
- catastrophizing,
- personalization,
- should statements,
- labeling.

Those patterns help the app explain *why* something was considered distorted.

## 5. How text becomes model input

Raw journal text cannot be sent directly into a neural network. It must first be converted into numbers.

The project uses a `TfidfVectorizer` saved inside `journal_analyzer_models.pkl`.

TF-IDF stands for term frequency-inverse document frequency. In simple terms, it turns each piece of text into a numeric vector that reflects which words appear and how important they are relative to the training data.

So the text:

> I always fail at everything

is transformed into a vector of numbers that the neural networks can process.

## 6. Loading the model files

The model loader is also in [model.py](model.py).

The file expects a pickle bundle named `journal_analyzer_models.pkl`.

That bundle is expected to contain:

- the trained TF-IDF vectorizer,
- the emotion label encoder,
- the saved weights for the distortion model,
- the saved weights for the emotion model.

When `load_models()` runs, it does the following:

1. Opens the pickle file.
2. Restores the vectorizer and emotion encoder.
3. Reconstructs the distortion network with the correct input size.
4. Loads the distortion weights.
5. Reconstructs the emotion network with the correct input size and number of emotion classes.
6. Loads the emotion weights.
7. Puts both models into evaluation mode.

If the pickle file is missing or loading fails, the function returns `False`.

## 7. Why `get_analyzer()` exists

`get_analyzer()` creates and caches one analyzer instance for the whole process.

That matters because:

- the model files should not be loaded on every request,
- the vectorizer and encoders should be reused,
- the app starts faster after the first load,
- request handling stays simpler.

This is a lazy singleton pattern:

- the first call loads the models and builds the analyzer,
- later calls reuse the same object.

## 8. What happens during analysis

The main inference method is `analyze_entry(text)`.

It performs these steps:

### Step 1: Vectorize the text

The input text is passed through the TF-IDF vectorizer.

### Step 2: Predict distortion

The distortion classifier produces one probability.

If the probability is greater than 0.5, the text is marked as distorted.

### Step 3: Predict emotion

The emotion classifier produces a score for each emotion class.

Those scores are converted into probabilities with softmax.

The emotion with the highest probability becomes the primary emotion.

### Step 4: Detect specific distortion patterns

The analyzer also checks the raw text against regex patterns.

This is useful because the neural model gives a score, but the regex rules can point to concrete examples like “always,” “never,” or “should.”

### Step 5: Build a health assessment

The analyzer combines distortion score, emotion, and detected patterns to produce a simple concern level:

- low,
- moderate,
- high.

It also generates a short recommendation.

## 9. What the analyzer returns

The returned analysis object contains:

- the original text,
- whether the text is considered distorted,
- the distortion probability,
- the detected distortion patterns,
- emotion probabilities,
- the primary emotion,
- the health assessment summary.

This makes the output useful both for machines and for humans reading the result.

## 10. How the Flask app uses the model

The API route in [routes/analyze.py](routes/analyze.py) receives JSON with a `text` field.

It sends the text to `compute_analysis_for_text()` in [services/analysis.py](services/analysis.py).

That service does two things:

### Rule-based analysis

It looks for explicit distortion phrases and positive patterns using regex.

This gives readable examples such as:

- all-or-nothing thinking,
- catastrophizing,
- gratitude,
- willingness to change.

### ML analysis

It calls `get_analyzer()` from [model.py](model.py) and runs the neural inference pipeline.

The service then combines the outputs into a single response with fields like:

- `distortions`
- `overallScore`
- `positivePatterns`
- `model`

## 11. What happens if the ML model fails

The service is designed to fail gracefully.

If `get_analyzer()` or model inference raises an exception, the app does not crash the request.

Instead, `compute_analysis_for_text()` falls back to a rule-based score derived from the distortion matches.

That means the application can still respond even if:

- `journal_analyzer_models.pkl` is missing,
- the model weights cannot be loaded,
- PyTorch inference fails for some reason.

## 12. Startup behavior

The Flask app in [app.py](app.py) tries to warm the analyzer when the server starts.

That means:

- the model is loaded once at startup if possible,
- the first user request is less likely to pay the loading cost,
- startup may take a little longer the first time the model files are loaded.

If warmup fails, the app prints a warning and still starts.

## 13. Important practical details

### This model is only as good as its training data

The distortion score and emotion label are learned from whatever data was used to train the saved weights.

That means the model can be wrong, especially on:

- short text,
- sarcasm,
- mixed emotions,
- unusual wording,
- domain-specific slang.

### The regex rules are not the same as the ML model

The rules are there to make the output easier to explain.

They do not replace the neural model.

In other words:

- the ML model gives a learned estimate,
- the rules give visible examples.

### The project is not using the notebook files at runtime

The notebooks in the repository may have been used during training or experimentation, but the production runtime logic is in [model.py](model.py) and [services/analysis.py](services/analysis.py).

## 14. Conceptual summary

If you want one mental model for the whole system, use this:

1. The text is turned into numbers with TF-IDF.
2. One neural network estimates whether the writing is cognitively distorted.
3. Another neural network estimates the emotion.
4. Regex rules highlight specific language patterns.
5. The service merges both views into a human-readable analysis response.

That is the core of the AI part of this project.
