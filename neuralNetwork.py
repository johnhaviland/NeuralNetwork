# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# Load NBA player dataset
# ADD PATH MANUALLY AFTER DOWNLOADING ACCOMPANYING FILE
nba_players_dataset = pd.read_json(r"C:\Users\jhavi\Downloads\nba-players_21-22.json")

input_features = [
    "player_height", "player_weight", "gp", "pts", "reb", "ast", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct"
]

# Define selection criteria
criteria = (
    (nba_players_dataset['gp'] >= 70) &
    (nba_players_dataset['player_height'] >= 194) &
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
y = nba_players_dataset['selected_for_team'].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the MLP model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile, train, and evaluate model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Filter the dataset based on criteria
filtered_players = nba_players_dataset[criteria]

# if there are players in the filtered dataset
if not filtered_players.empty:
    # Extract input features and target from the filtered dataset
    filtered_x_test = filtered_players[input_features].values
    filtered_y_test = filtered_players['selected_for_team'].values

    # Predict labels for the filtered dataset
    filtered_predictions = model.predict(filtered_x_test)

    # Sort the players based on predicted probability
    top_players_indices = np.argsort(filtered_predictions[:, 0])[::-1]
    top_players = filtered_players.iloc[top_players_indices[:5]]
    print("Optimal Team:")
    print(top_players)
else:
    print("No players meet the criteria.")

