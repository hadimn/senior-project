## 📧 Email Spam Detection Web App (Senior Project)

This Python-based application detects whether email messages are spam by leveraging Natural Language Processing (NLP) and Machine Learning (ML).

### 🎯 Purpose & Overview

* **Goal:** Automatically classify emails into “spam” or “not spam”.
* **Approach:**

  1. Preprocess raw email text.
  2. Extract meaningful features (e.g., token frequency, TF-IDF).
  3. Train ML classifiers (e.g., Naive Bayes, Logistic Regression).
  4. Evaluate accuracy, precision, recall, and F1-score on a validation set.

### 📁 Repository Structure

```
├── data/                  # Raw and preprocessed datasets (e.g., spam vs. ham)
├── notebooks/             # Jupyter notebooks for model development
│   ├── exploration.ipynb  # EDA, text analysis, dataset cleaning
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb   # Metrics, ROC curves, confusion matrices
├── src/                   # Optional scripts (.py) for preprocessing and model pipelines
├── requirements.txt       # Dependencies (e.g., scikit-learn, pandas, nltk)
└── README.md              # Project overview and usage guide
```

### 🛠️ Setup & Execution

1. **Clone the repository**

   ```bash
   git clone https://github.com/hadimn/senior-project.git
   cd senior-project
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```
4. **Execute notebooks** in the order:

   * `exploration.ipynb`
   * `feature_engineering.ipynb`
   * `model_training.ipynb`
   * `evaluation.ipynb`

### 📊 What You’ll Learn

* Data cleaning (text normalization, stop-word removal, tokenization)
* Feature extraction (bag-of-words, TF-IDF)
* ML model selection and training (e.g., Naive Bayes, Logistic Regression)
* Performance evaluation using standard metrics
* Optional enhancements: ensemble models, deep learning approaches

### ✅ Next Steps / Enhancements

* Convert notebooks into reusable Python modules for production pipelines
* Add a Flask or FastAPI wrapper for live email input and classification
* Improve performance with advanced NLP techniques (word embeddings, transformer-based models)
* Create a front-end interface capable of ingesting real emails for predictions
