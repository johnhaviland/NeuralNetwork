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
    features = [player['ast'], player['reb'], player['gp']] 
    X_train.append(features)

    label = [1, 0, 0, 0, 0]
    
    # Extract labels based on some criteria
    if player['ast'] > 0.4:  # Just an example criterion
        label = [1, 0, 0, 0, 0]
    elif player['reb'] + player['gp'] > 100:  # Another example criterion
        label = [0, 1, 0, 0, 0]
    # ... Add more criteria for other roles
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3,)),  # Input layer
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
    tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer 2
    tf.keras.layers.Dense(5, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

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
    print(player['player_name'])  # Assuming each player has a 'name' attribute in the JSON data