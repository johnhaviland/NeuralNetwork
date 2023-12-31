import json
import numpy as np
import tensorflow as tf

# Load the JSON data
with open('nba-players_21-22.json', 'r') as file:
    data = json.load(file)

# Filter players based on a 5-year window, e.g., players drafted between 2015 and 2020
# selected_players = [player for player in data if 2015 <= player['draft_year'] <= 2020]

# Select a pool of 100 players from the filtered list
pool_of_players = data

# Data Preparation
X_train = []
y_train = []

for player in pool_of_players:
    # Extract features
    features = [player['player_height'], player['gp'], player['pts'], player['reb'], player['ast'], player['net_rating'], player['oreb_pct'], player['dreb_pct'], player['usg_pct'], player['ts_pct'], player['ast_pct']]
    X_train.append(features)

    position = [0, 0, 0, 0, 0]      # initializing position array

    # Extract position labels based on some criteria
    if player['player_height'] > 180 and player['gp'] > 60 and player['ast'] > 6 and player['net_rating'] > 0 and player['usg_pct'] < 0.35 and player['ts_pct'] > 0.57 and player['ast_pct'] > 0.15:
        position = [1, 0, 0, 0, 0]  # point guard
    elif player['player_height'] > 195 and player['gp'] > 60 and player['pts'] > 18 and player['net_rating'] > 0 and player['usg_pct'] < 0.35 and player['ts_pct'] > 0.57 and player['ast_pct'] > 0.1:
        position = [0, 1, 0, 0, 0]     # shooting guard
    elif player['player_height'] > 203 and player['gp'] > 60 and player['pts'] > 18 and player['reb'] > 4 and player['net_rating'] > 0 and player['usg_pct'] < 0.35 and player['ts_pct'] > 0.57 and player['dreb_pct'] > 0.12 and player['ast_pct'] > 0.1:
        position = [0, 0, 1, 0, 0]     # small forward
    elif player['player_height'] > 205 and player['gp'] > 60 and player['pts'] > 18 and player['reb'] > 6 and player['net_rating'] > 0 and player['usg_pct'] < 0.35 and player['ts_pct'] > 0.57 and player['dreb_pct'] > 0.16 and player['ast_pct'] > 0.1:
        position = [0, 0, 0, 1, 0]     # power forward
    elif player['player_height'] > 208 and player['gp'] > 60 and player['reb'] > 6 and player['net_rating'] > 0 and player['usg_pct'] < 0.35 and player['ts_pct'] > 0.57 and player['dreb_pct'] > 0.22 and player['ast_pct'] > 0.05:
        position = [0, 0, 0, 0, 1]     # center
    y_train.append(position)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(11,)),  # Input layer
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer 2
    tf.keras.layers.Dense(5, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict using the model
predictions = model.predict(X_train)

# Get the indices of the players with the highest probability for each role without duplication
selected_indices = set()
optimal_team_indices = []

for role_probs in predictions.T:  # Transpose to iterate over roles
    sorted_indices = np.argsort(role_probs)[::-1]  # Sort indices by descending probability
    for index in sorted_indices:
        if index not in selected_indices:
            optimal_team_indices.append(index)
            selected_indices.add(index)
            break

# Extract the optimal team from the pool_of_players using the indices
optimal_team = [pool_of_players[index] for index in optimal_team_indices]

# Print the optimal team
for player in optimal_team:
    print(player['player_name'], )  # Assuming each player has a 'name' attribute in the JSON data
