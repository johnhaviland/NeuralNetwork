# Required Libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from nltk.tokenize import word_tokenize
import string
import nltk
import pandas

nba_players_dataset = pandas.read_csv(r"C:\Users\jhavi\OneDrive\Documents\nba-players_21-22.csv")

print(nba_players_dataset)

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Load Dataset
data = fetch_20newsgroups(subset='all')
X, y = data.data, data.target


# Preprocess text data: Convert to lowercase, remove punctuation, tokenize
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    return ' '.join(tokens)


X_preprocessed = [preprocess_text(text) for text in X]


# Convert Text Data to Numerical Format
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X_preprocessed)
X_vectorized = X_vectorized.toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)

# Design Neural Network Architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=X_train.shape[1]))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
