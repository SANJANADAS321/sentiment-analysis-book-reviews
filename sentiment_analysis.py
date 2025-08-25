# Sentiment Analysis on Book Reviews
# With Word Clouds, Prediction Logging, and Sentiment Pie Chart

import csv
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# STEP 1 — Read CSV & Clean
# -------------------------
reviews = []
labels = []

with open(r"C:\Users\sanja\OneDrive\Sentiment_Analysis(python+ML)\book_reviews.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            text = row[0].strip()
            sentiment = row[1].strip().lower()
            labels.append(sentiment)
            reviews.append(text)

# -------------------------
# STEP 2 — Filter out 'neutral'
# -------------------------
filtered_reviews = []
filtered_labels = []

for review, sentiment in zip(reviews, labels):
    if sentiment in ["positive", "negative"]:  # keep only 2 classes
        filtered_reviews.append(review)
        filtered_labels.append(sentiment)

reviews = filtered_reviews
labels = filtered_labels

print("Unique sentiments:", set(labels))
print("Number of positive reviews:", labels.count("positive"))
print("Number of negative reviews:", labels.count("negative"))

# -------------------------
# STEP 3 — Train/Test Split
# -------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# STEP 4 — Train Model
# -------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------
# STEP 5 — Evaluate
# -------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.1f} %")

# -------------------------
# STEP 6 — Visualizations
# -------------------------
df = pd.DataFrame({"review": reviews, "sentiment": labels})

# Sentiment count plot
df["sentiment"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Word Clouds
positive_text = " ".join(df[df["sentiment"] == "positive"]["review"])
negative_text = " ".join(df[df["sentiment"] == "negative"]["review"])

if positive_text.strip():
    WordCloud(width=500, height=300, background_color="white").generate(positive_text).to_file("positive_wc.png")
    plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(positive_text))
    plt.axis("off")
    plt.title("Positive Reviews Word Cloud")
    plt.show()

if negative_text.strip():
    WordCloud(width=500, height=300, background_color="white").generate(negative_text).to_file("negative_wc.png")
    plt.imshow(WordCloud(width=500, height=300, background_color="white").generate(negative_text))
    plt.axis("off")
    plt.title("Negative Reviews Word Cloud")
    plt.show()

# -------------------------
# STEP 7 — User Input
# -------------------------
while True:
    user_review = input("Type a book review to check sentiment (type 'quit' to exit): ")
    if user_review.lower() == "quit":
        break
    user_vector = vectorizer.transform([user_review])
    prediction = model.predict(user_vector)[0]
    print(f"Predicted Sentiment: {prediction}")
