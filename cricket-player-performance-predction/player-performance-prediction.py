import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load historical data on the player's performance
# Sample data is already added to the csv file
df = pd.read_csv('./player_stats.csv')

# Preprocess the data by removing missing values and selecting relevant features
df = df.dropna()
stats_to_predicte_score = df[['BattingAverage', 'StrikeRate', 'WicketsTaken']].values

stats_to_predicte_wickets = df[['BattingAverage', 'StrikeRate', 'Score']].values

score_data = df['Score'].values

wickets_data = df['WicketsTaken'].values

# Train a random forest regression model on the historical data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(stats_to_predicte_score, score_data)

# Get the player's statistics for the upcoming match
upcoming_match_data = [[45.2, 78.6, 5]]  # Example input features "Batting Average", "Strike Rate",
upcoming_match_data = np.array(upcoming_match_data)

# Make a prediction using the trained model
predicted_score = model.predict(upcoming_match_data)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(stats_to_predicte_wickets, wickets_data)

upcoming_match_data = [[45.2, 78.6, 112]]  # Example input features
upcoming_match_data = np.array(upcoming_match_data)
predicted_wickets = model.predict(upcoming_match_data)


print(f'Predicted score & wickets for the upcoming match: {float(predicted_score)}, {int(predicted_wickets)}')
