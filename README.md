# STR Sentiment Analysis: VADER vs ROBERTA

A sentiment analysis project on **short-term rental (STR) reviews** comparing a **rule-based model (VADER)** with a **transformer-based model (ROBERTA)**.

**Kaggle notebook:** <https://www.kaggle.com/code/samio23/str-sentiment-analysis-vader-vs-roberta>

---

## Overview

This project analyzes customer reviews from short-term rental listings and classifies their sentiment as **positive**, **neutral**, or **negative**.

The notebook compares two different NLP approaches:

- **VADER** — a lightweight, lexicon- and rule-based sentiment analyzer
- **ROBERTA** — a deep learning model that captures context and nuance more effectively

The main goal is to see how much performance and interpretability differ when moving from a simple sentiment model to a contextual transformer model on real-world review data.

---

## Objectives

- Load and inspect STR review data
- Handle **multilingual reviews** by translating them into English
- Apply **VADER** sentiment analysis
- Apply **ROBERTA** sentiment analysis using `cardiffnlp/twitter-roberta-base-sentiment`
- Compare the output of both models on the same reviews
- Explore how sentiment scores relate to review ratings

---

## Dataset

The notebook uses an STR reviews CSV file from Kaggle and keeps the following columns:

- `guest_first_name`
- `rating`
- `public_review`

From the notebook output:

- **Rows:** 238
- **Columns used:** 3
- **Missing values:** None in the selected columns

---

## Workflow

### 1. Data Loading
The dataset is loaded from Kaggle and reduced to the most relevant columns for analysis.

### 2. Translation to English
Because the reviews contain multiple languages, the notebook translates review text into English using **GoogleTranslator** from `deep_translator`.

This makes the inputs more consistent for downstream sentiment models, especially VADER.

### 3. Exploratory Data Analysis
The notebook includes:
- A quick preview of the dataset
- Dataset shape and null-checks
- A bar plot of review rating distribution

### 4. VADER Sentiment Analysis
VADER is used to generate:
- `neg`
- `neu`
- `pos`
- `compound`

These scores are merged back into the main review dataframe for comparison with review ratings.

### 5. ROBERTA Sentiment Analysis
The notebook uses the Hugging Face model:

`cardiffnlp/twitter-roberta-base-sentiment`

For each review, the model outputs:
- `roberta_neg`
- `roberta_neu`
- `roberta_pos`

### 6. Model Comparison
The final section compares VADER and ROBERTA scores side by side on randomly selected reviews to highlight where contextual understanding changes the interpretation.

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **NLTK / VADER**
- **Transformers (Hugging Face)**
- **SciPy**
- **tqdm**
- **deep_translator**

---

## Installation

Install the required packages before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers scipy tqdm deep_translator torch
