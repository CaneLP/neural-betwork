import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import glob
# Beautify print - delete later
import sys
# import keras.backend as K

np.set_printoptions(threshold=sys.maxsize)

path = '../data'

files = [f for f in glob.glob(path + "**/*.csv", recursive=True)]

matches = pd.DataFrame()

for f in files:
    fields = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
    curr_season = pd.read_csv(f, error_bad_lines=False, usecols=fields)
    curr_season.dropna(inplace=True)
    curr_season['FTHG'] = curr_season['FTHG'].astype(int)
    curr_season['FTAG'] = curr_season['FTAG'].astype(int)
    matches = matches.append(curr_season, ignore_index=True, sort='False')

last_n_games = 10

#TODO
# separate data in class Team
# add more data maybe

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
rows_to_drop = []
for key, value in match.items():
    if np.count_nonzero(match[key][1]) == 0 or np.count_nonzero(match[key][2]) == 0:
        # print(key, value)
        rows_to_drop.append(key)
    else:
        matches_nn_input.append(match[key][1][1:] + match[key][2][1:])

matches = matches.drop(rows_to_drop)
matches.index = range(len(matches))
matches_nn_input = np.array(matches_nn_input)

output_class = ['H', 'D', 'A']

full_time_results = matches['FTR']
output_final_ints = []
for res in full_time_results:
    if res == 'H':
        output_final_ints.append(1)
    elif res == 'A':
        output_final_ints.append(2)
    else:
        output_final_ints.append(0)
output_final_ints = np.array(output_final_ints)

train_input, test_input, train_output, test_output =\
    train_test_split(matches_nn_input, output_final_ints, test_size=0.3, shuffle=False)


# Normalized input
max_col_values_train = [max(l) for l in list(zip(*train_input))]
print("max column training values:", max_col_values_train)

train_input = [list(zip(line, max_col_values_train)) for line in train_input]
train_input = [[t[0]/t[1] for t in line] for line in train_input]
train_input = np.array(train_input)
# print(train_input)

max_col_values_test = [max(l) for l in list(zip(*test_input))]
print("max column test values:", max_col_values_test)
test_input = [list(zip(line, max_col_values_test)) for line in test_input]
test_input = [[t[0]/t[1] for t in line] for line in test_input]
test_input = np.array(test_input)
# print(test_input)

hidden_layer_1 = 100
hidden_layer_2 = 50

model = keras.Sequential([keras.layers.Dense(hidden_layer_1, input_shape=(matches_nn_input.shape[1], ), activation=tf.nn.relu),
                          keras.layers.Dense(hidden_layer_2, activation=tf.nn.relu),
                          keras.layers.Dense(len(output_class), activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_input, train_output, epochs=10)

print("Testing...")
test_loss, test_acc = model.evaluate(test_input, test_output)
print('Test accuracy:', test_acc)

prediction = model.predict(test_input)

# for i in range(21):
#     print(test_output[i])
#     print(np.argmax(prediction[i]))
#     print("-----------------------")

