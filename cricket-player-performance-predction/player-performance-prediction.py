import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load historical data on the player's performance
df = pd.read_csv('./player_stats.csv')

# Preprocess the data by removing missing values and selecting relevant features
df = df.dropna()
X = df[['BattingAverage', 'StrikeRate', 'WicketsTaken']].values
Z = df[['BattingAverage', 'StrikeRate', 'Score']].values
y = df['Score'].values
A = df['WicketsTaken'].values

# Train a random forest regression model on the historical data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get the player's statistics for the upcoming match
upcoming_match_data = [[45.2, 78.6, 5]]  # Example input features
upcoming_match_data = np.array(upcoming_match_data)

# Make a prediction using the trained model
predicted_score = model.predict(upcoming_match_data)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(Z, A)

upcoming_match_data = [[45.2, 78.6, 112]]  # Example input features
upcoming_match_data = np.array(upcoming_match_data)
predicted_wickets = model.predict(upcoming_match_data)


print(f'Predicted score & wickets for the upcoming match: {float(predicted_score)}, {int(predicted_wickets)}')
