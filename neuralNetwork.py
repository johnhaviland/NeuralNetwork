# Required Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import string
# import nltk
import pandas as pd

# Load NBA Players Dataset
nba_players_dataset = pd.read_json(r"nba-players_21-22.json")

# features
features = ['ast', 'reb', 'gp', 'usg_pct', 'dreb_pct']
X = nba_players_dataset[features]


# Define the selection criteria
criteria = (
    (nba_players_dataset['gp'] >= 75) &
    (nba_players_dataset['player_height'] >= 196) &
    (nba_players_dataset['reb'] >= 4) &
    (nba_players_dataset['ast_pct'] >= 0.1) &
    (nba_players_dataset['ts_pct'] >= 0.56)
)

# Create the 'selected_for_team' column based on the criteria
nba_players_dataset['selected_for_team'] = criteria.astype(int)


# Extract the input features from the dataset

# Target variable
y = nba_players_dataset['selected_for_team'].values  # 'selected_for_team' is the target variable

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training
X_train, X_test, y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=11)

# Define the MLP model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Input layer
model.add(Dense(64, activation='relu'))  # Hidden layer(s)
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')


predictions = model.predict(X_test)
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
print("Criterion 4 (reb >= 6):", sum(nba_players_dataset['reb'] >= 6))
print("Criterion 5 (ast_pct >= 0.2):", sum(nba_players_dataset['ast_pct'] >= 0.2))
print("Criterion 7 (ts_pct >= 0.58):", sum(nba_players_dataset['ts_pct'] >= 0.58))
