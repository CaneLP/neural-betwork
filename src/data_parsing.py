import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from tensorflow.python.keras import activations
# Beautify print - delete later
import sys
np.set_printoptions(threshold=sys.maxsize)

#TODO take all files from folder data in the loop
season_1 = pd.read_csv('../data/season_16-17.csv')
season_2 = pd.read_csv('../data/season_17-18.csv')
season_3 = pd.read_csv('../data/season_18-19.csv')

matches = season_1.append(season_2, ignore_index=True)
matches = matches.append(season_3, sort='False', ignore_index=True)
# print(matches)

last_n_games = 10

#TODO
# separate this data in class Team
# add more data maybe
# ah = at_home

# home_matches = 0
# home_wins_ah = 0
# home_wins_total = 0
# home_draws_ah = 0
# home_draws_total = 0
# home_loses_ah = 0
# home_loses_total = 0
# home_goals_ah = 0
# home_goals_total = 0
# home_goals_conceded_ah = 0
# home_goals_conceded_total = 0
# home_h2h_wins_ah = 0
# home_h2h_wins_total = 0
# home_h2h_goals_scored_ah = 0
# home_h2h_goals_conceded_total = 0

# away_matches = 0
# away_wins_ah = 0
# away_wins_total = 0
# away_draws_ah = 0
# away_draws_total = 0
# away_loses_ah = 0
# away_loses_total = 0
# away_goals_scored_ah = 0
# away_goals_total = 0
# away_goals_conceded_ah = 0
# away_goals_conceded_total = 0

#Collecting data
match = {}
team = {}
for i in range(len(matches)):
    match_index = len(matches) - i - 1
    curr_match = matches.iloc[len(matches) - i - 1]
    match[match_index] = [(curr_match['HomeTeam'], curr_match['AwayTeam']), [0] * 6, [0] * 6]

    if curr_match['HomeTeam'] in team:
        team[curr_match['HomeTeam']] = team[curr_match['HomeTeam']] + [match_index]
    else:
        team[curr_match['HomeTeam']] = [match_index]

    if curr_match['AwayTeam'] in team:
        team[curr_match['AwayTeam']] = team[curr_match['AwayTeam']] + [match_index]
    else:
        team[curr_match['AwayTeam']] = [match_index]

    if len(team[curr_match['HomeTeam']]) > last_n_games:
        curr_team = copy.deepcopy(curr_match['HomeTeam'])
        match_key = copy.deepcopy(team[curr_team].pop(0))
        curr_team_matches = copy.deepcopy(team[curr_team])

        games_total = 0
        wins_total = 0
        draws_total = 0
        losses_total = 0
        goals_scored_total = 0
        goals_conceded_total = 0

        for key in curr_team_matches:
            games_total += 1
            match_calc = matches.iloc[key]
            if match_calc['FTR'] == 'H' and match_calc['HomeTeam'] == curr_team:
                wins_total += 1
            elif match_calc['FTR'] == 'A' and match_calc['AwayTeam'] == curr_team:
                wins_total += 1

            if match_calc['FTR'] == 'H' and match_calc['AwayTeam'] == curr_team:
                losses_total += 1
            elif match_calc['FTR'] == 'A' and match_calc['HomeTeam'] == curr_team:
                losses_total += 1

            if match_calc['FTR'] == 'D':
                draws_total += 1

            if match_calc['HomeTeam'] == curr_team:
                goals_scored_total += match_calc['FTHG']
                goals_conceded_total += match_calc['FTAG']
            elif match_calc['AwayTeam'] == curr_team:
                goals_scored_total += match_calc['FTAG']
                goals_conceded_total += match_calc['FTHG']

        home_or_away = 2
        if matches.iloc[match_key]['HomeTeam'] == curr_team:
            home_or_away = 1
        match[match_key][home_or_away] = [games_total, wins_total, draws_total,
                                          losses_total, goals_scored_total, goals_conceded_total]
    if len(team[curr_match['AwayTeam']]) > last_n_games:
        curr_team = copy.deepcopy(curr_match['AwayTeam'])
        match_key = copy.deepcopy(team[curr_team].pop(0))
        curr_team_matches = copy.deepcopy(team[curr_team])

        games_total = 0
        wins_total = 0
        draws_total = 0
        losses_total = 0
        goals_scored_total = 0
        goals_conceded_total = 0

        for key in curr_team_matches:
            games_total += 1
            match_calc = matches.iloc[key]

            if match_calc['FTR'] == 'H' and match_calc['HomeTeam'] == curr_team:
                wins_total += 1
            elif match_calc['FTR'] == 'A' and match_calc['AwayTeam'] == curr_team:
                wins_total += 1

            if match_calc['FTR'] == 'H' and match_calc['AwayTeam'] == curr_team:
                losses_total += 1
            elif match_calc['FTR'] == 'A' and match_calc['HomeTeam'] == curr_team:
                losses_total += 1

            if match_calc['FTR'] == 'D':
                draws_total += 1

            if match_calc['HomeTeam'] == curr_team:
                goals_scored_total += match_calc['FTHG']
                goals_conceded_total += match_calc['FTAG']
            elif match_calc['AwayTeam'] == curr_team:
                goals_scored_total += match_calc['FTAG']
                goals_conceded_total += match_calc['FTHG']

        home_or_away = 2
        if matches.iloc[match_key]['HomeTeam'] == curr_team:
            home_or_away = 1
        match[match_key][home_or_away] = [games_total, wins_total, draws_total,
                                          losses_total, goals_scored_total, goals_conceded_total]

# for key, value in match.items():
#         print(key, value)

# Store only the data where statistics exists for future use - profit calculation
# Create an array with the data for the neural network input
matches_nn_input = []
for key, value in match.items():
    rows_to_drop = []
    if np.count_nonzero(match[key][1]) == 0 or np.count_nonzero(match[key][2]) == 0:
        # print(key, value)
        rows_to_drop.append(key)
    else:
        matches_nn_input.append(match[key][1][1:] + match[key][2][1:])

matches = matches.drop(rows_to_drop)
matches_nn_input = np.array(matches_nn_input)
# print(matches_nn_input)
# print(matches)

exit()

home_teams = matches['HomeTeam']
away_teams = matches['AwayTeam']
home_team_goals = matches['FTHG']
away_team_goals = matches['FTAG']

all_teams = {}
team_id = 0
for team in home_teams:
    if team not in all_teams:
        all_teams[team] = team_id
        team_id = team_id + 1

# print(all_teams)
output_class = ['H', 'D', 'A']

output = matches['FTR']

output_binary_results = []
for res in output:
    if res == 'H':
        output_binary_results.append([1, 0, 0])
    elif res == 'A':
        output_binary_results.append([0, 0, 1])
    else:
        output_binary_results.append([0, 1, 0])

# print(output_final)

dataset = [[all_teams[home_teams[i]],
            all_teams[away_teams[i]],
            home_team_goals[i],
            away_team_goals[i]]
           for i in range(len(matches))]
# print(dataset)

for i in range(len(dataset)):
    dataset[i] = dataset[i] + output_binary_results[i]

# print(dataset)

hidden_layer_1 = 50
hidden_layer_2 = 25
output = [res for res in output]
# print(output)
dataset = np.array(dataset)
output_final = np.array(output)

# print(dataset.shape[0])
# print(output_final.shape[0])
#
# print(len(dataset))
# print(len(output_class))
# print(dataset)
# print(output_final)
# exit()

output_final_ints = []
for res in output_final:
    if res == 'H':
        output_final_ints.append(1)
    elif res == 'A':
        output_final_ints.append(2)
    else:
        output_final_ints.append(0)
output_final_ints = np.array(output_final_ints)

train_input, test_input, train_output, test_output =\
    train_test_split(dataset, output_final_ints, test_size=0.2, shuffle=False)

# print(train_input.shape)
# print(train_output.shape)
# print(train_input)
# print(train_output)
# print(len(train_input))
# print(len(train_output))

# exit()

model = keras.Sequential([keras.layers.Dense(7),
                          keras.layers.Dense(hidden_layer_1, activation=tf.nn.relu),
                          keras.layers.Dense(hidden_layer_2, activation=tf.nn.relu),
                          keras.layers.Dense(len(output_class), activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_input, train_output, epochs=5)

test_loss, test_acc = model.evaluate(test_input, test_output)
print('Test accuracy:', test_acc)

prediction = model.predict(test_input)

# for i in range(21):
#     print(test_output[i])
#     print(np.argmax(prediction[i]))
#     print("-----------------------")

