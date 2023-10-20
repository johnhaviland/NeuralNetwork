import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


import nltk
nltk.download('stopwords')

# Load your data into a pandas dataframe
data = pd.read_csv('hate_speech_dataset.csv', names=['tweet', 'sentiment'], header=0)

# Handle missing values
data = data.dropna()

# Remove user @'s and non alphabetical characters
data['tweet'] = data['tweet'].replace(to_replace=r'@\w+', value='', regex=True)
data['tweet'] = data['tweet'].replace(to_replace='[^A-Za-z0-9\s]+', value='', regex=True)

# Remove punctuation and stop words
def clean_text(text):
    text = str(text)
    text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)

data['tweet'] = data['tweet'].apply(clean_text)
data.head()

# print(data['tweet'])

# Assign each word in every text element a sentiment score using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['tweet'])
y = data['sentiment']

# Divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use a binary classification algorithm such as logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Compute the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_data = ["I hate my life", "I love you", "white supremacy", "kill everyone"]

clean_new_data = clean_text(new_data)

# Transform the new data using the TfidfVectorizer
X_new = vectorizer.transform(new_data)

# Predict the sentiment using the trained model
predictions = clf.predict(X_new)

print(predictions)
