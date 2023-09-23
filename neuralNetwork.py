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
import pandas as pd

# Load NBA Players Dataset
nba_players_dataset = pd.read_json(r"C:\Users\jhavi\Downloads\nba-players_21-22.json")

input_features = [
    "player_height", "player_weight", "gp", "pts", "reb", "ast", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"
]


# Define the selection criteria
criteria = (
    (nba_players_dataset['gp'] >= 75) &
    (nba_players_dataset['player_height'] >= 196) &
    (nba_players_dataset['pts'] >= 15) &
    (nba_players_dataset['reb'] >= 4) &
    (nba_players_dataset['ast_pct'] >= 0.1) &
    (nba_players_dataset['ts_pct'] >= 0.56)
)

# Create the 'selected_for_team' column based on the criteria
nba_players_dataset['selected_for_team'] = criteria.astype(int)


# Extract the input features from the dataset
x = nba_players_dataset[input_features].values

# Target variable
y = nba_players_dataset['selected_for_team'].values  # 'selected_for_team' is the target variable

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the MLP model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))  # Input layer
model.add(Dense(64, activation='relu'))  # Hidden layer(s)
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2)

# Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')


predictions = model.predict(x_test)
threshold = 0.5
predicted_labels = (predictions > threshold).astype(int)

# Select the top 5 players with the highest predicted probability
top_players_indices = predicted_labels[:, 0].argsort()[-5:][::-1]
top_players = nba_players_dataset.iloc[top_players_indices]
print("Optimal Team:")
print(top_players)

# Check the number of players meeting each criterion
print("Number of players meeting each criterion:")
print("Criterion 1 (gp >= 75):", sum(nba_players_dataset['gp'] >= 75))
print("Criterion 2 (player_height >= 200):", sum(nba_players_dataset['player_height'] >= 200))
print("Criterion 3 (pts >= 16):", sum(nba_players_dataset['pts'] >= 16))
print("Criterion 4 (reb >= 6):", sum(nba_players_dataset['reb'] >= 6))
print("Criterion 5 (ast_pct >= 0.2):", sum(nba_players_dataset['ast_pct'] >= 0.2))
print("Criterion 7 (ts_pct >= 0.58):", sum(nba_players_dataset['ts_pct'] >= 0.58))
