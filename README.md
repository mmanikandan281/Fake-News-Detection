# 📰 Fake News Detection using NLP & ML

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Colab-green?logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## ▶️ Run on Google Colab

Click the badge below to run this project in Colab:

[![Open In Colab](https://colab.research.google.com/drive/1eErFvKMsueFHUFQCk__tDqyF-Bcf8-Xh?usp=sharing)

## 📌 Project Overview

**Fake News Detection** is a Natural Language Processing (NLP) project that classifies news articles as **Fake 🟥** or **Real 🟩**.  
This is done using machine learning algorithms and text vectorization techniques (TF-IDF, Count Vectorizer).  
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 🧠 Algorithms & Techniques Used

- 🔤 **Text Preprocessing**: Tokenization, Lemmatization, POS Tagging
- 🧼 **Cleaning**: Removal of stopwords, punctuation
- 🧾 **Feature Engineering**: CountVectorizer, TF-IDF
- 🧪 **Model Training**: Logistic Regression, SVM, Naive Bayes, Decision Tree, Random Forest, XGBoost
- 📊 **Evaluation**: Accuracy Score using GridSearchCV

---

## 📁 Dataset

- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

| Column Name | Description       |
|-------------|-------------------|
| title       | Headline of article |
| text        | Full content of article |
| subject     | Category (politics, etc.) |
| date        | Published date         |

---

## 🛠️ How It Works

<details>
<summary><strong>📘 Step-by-step Pipeline</strong></summary>

1. **Mount Google Drive** and Load Dataset  
2. **Data Cleaning & Preprocessing**
   - Lowercasing, removing punctuation
   - POS tagging and lemmatization  
3. **Visualization**  
   - WordClouds, Bar plots
4. **Vectorization**  
   - Using CountVectorizer and TF-IDF
5. **Train/Test Split**  
6. **Model Training**  
   - Hyperparameter tuning using `GridSearchCV`
7. **Prediction**  
   - Predict label for new unseen article
</details>

---

## 📊 Visualizations

| WordCloud - Fake News | WordCloud - Real News |
|------------------------|------------------------|
| ![image](https://github.com/user-attachments/assets/a89cbff0-9d7e-493c-a24e-7317cbfcc340) | ![image](https://github.com/user-attachments/assets/d6952bb2-262f-4041-8063-4abfcd9a8a15) |

---

## 🤖 Best Performing Model

| Model        | Accuracy (TF-IDF) |
|--------------|-------------------|
| Decision Tree | ✅ **94.2%**       |

---

## 💬 Example Prediction

<details>
<summary><strong>Click to see test prediction</strong></summary>

```python
text = ["Donald Trump just couldn’t wish all Americans a Happy New Year and leave it at that..."]
# Preprocess -> Lemmatize -> Vectorize -> Predict
result = model.predict(tfidf_vectorizer.transform([processed_text]))
print(result)
