# Sentiment Analysis on Book Reviews
# With Word Clouds, Prediction Logging, and Sentiment Pie Chart

import string
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ----------------------------
# Step 1: Read dataset from CSV
# ----------------------------
reviews = []
labels = []

with open("book_reviews.csv", "r", encoding="utf-8") as f:
    for line in f:
        text, sentiment = line.strip().split(",")
        reviews.append(text)
        labels.append(sentiment)

# ----------------------------
# Step 2: Clean text data
# ----------------------------
cleaned_reviews = []
for review in reviews:
    review = review.lower()
    review = review.translate(str.maketrans("", "", string.punctuation))
    cleaned_reviews.append(review)

# ----------------------------
# Step 3: Convert text to numbers
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_reviews)

# ----------------------------
# Step 4: Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# ----------------------------
# Step 5: Train Naive Bayes model
# ----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ----------------------------
# Step 6: Evaluate model
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Model Accuracy:", round(accuracy, 2), "%")

# ----------------------------
# Step 7: Word Clouds
# ----------------------------
positive_text = " ".join(
    [cleaned_reviews[i] for i in range(len(cleaned_reviews)) if labels[i] == "positive"]
)
negative_text = " ".join(
    [cleaned_reviews[i] for i in range(len(cleaned_reviews)) if labels[i] == "negative"]
)

positive_wc = WordCloud(width=500, height=300, background_color="white").generate(positive_text)
negative_wc = WordCloud(width=500, height=300, background_color="white").generate(negative_text)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(positive_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Word Cloud")

plt.subplot(1, 2, 2)
plt.imshow(negative_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Word Cloud")

plt.show()

# ----------------------------
# Step 8: Interactive prediction + logging
# ----------------------------
log_data = []

print("\nType a book review to check sentiment (type 'quit' to exit):")
while True:
    user_review = input("> ")
    if user_review.lower() == "quit":
        break
    cleaned = user_review.lower().translate(str.maketrans("", "", string.punctuation))
    X_new = vectorizer.transform([cleaned])
    prediction = model.predict(X_new)[0]
    print("Predicted Sentiment:", prediction)

    # Store in log
    log_data.append({"review": user_review, "predicted_sentiment": prediction})

# ----------------------------
# Step 9: Save predictions to CSV
# ----------------------------
if log_data:
    df = pd.DataFrame(log_data)
    df.to_csv("predictions_log.csv", index=False)
    print("\nPredictions saved to predictions_log.csv")

    # ----------------------------
    # Step 10: Pie chart of predictions
    # ----------------------------
    sentiment_counts = df["predicted_sentiment"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["#66bb6a", "#ef5350", "#ffee58"])
    plt.title("Sentiment Distribution from This Session")
    plt.show()
