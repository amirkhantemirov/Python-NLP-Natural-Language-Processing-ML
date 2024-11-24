
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import pipeline


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


data = pd.DataFrame({
    'aspect': ['service', 'food', 'ambience', 'price'],
    'review': [
        'The service was excellent, and staff were friendly',
        'The food was not fresh and tasted bad',
        'Ambience was relaxing, we enjoyed our time there',
        'The price was too high for the quality of the food'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
})


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  
    return ' '.join(tokens)

data['processed_review'] = data['review'].apply(preprocess_text)


X = data['processed_review']
y = data['sentiment']


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Классификация есебі:")
print(classification_report(y_test, y_pred))


new_reviews = [
    "The service was slow and disappointing",
    "The food was delicious and fresh",
    "The ambience was chaotic and noisy"
]
new_reviews_processed = [preprocess_text(review) for review in new_reviews]
new_reviews_vectorized = vectorizer.transform(new_reviews_processed)
predictions = model.predict(new_reviews_vectorized)


for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review} -> Sentiment: {sentiment}")


aspect_sentiment_analyzer = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
example = "The price was reasonable and the service was fantastic."
print(aspect_sentiment_analyzer(example))
