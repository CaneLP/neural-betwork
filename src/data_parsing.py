import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import activations

#TODO take all files from folder data in the loop
season_1 = pd.read_csv('../data/season_16-17.csv')
season_2 = pd.read_csv('../data/season_17-18.csv')
season_3 = pd.read_csv('../data/season_18-19.csv')

matches = season_1.append(season_2, ignore_index=True)
matches = matches.append(season_3, sort='False', ignore_index=True)
# print(matches)
#
# exit()

#TODO write function for this
last_n_games = 20

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
# TODO h2h match not existed problem, should we add h2h at all?
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
i = 0

match = {}
team = {}
# print("----------------------")
# print(matches.iloc[1])
# print("----------------------")
# exit()
# for i in range(len(matches)):
for i in range(50):
    #count previous games of that home/away team
    count_home_matches = 0
    count_away_matches = 0

    match_index = len(matches) - i
    curr_match = matches.iloc[len(matches) - i - 1]
    home_matches = 0
    # home_wins_ah = 0
    home_wins_total = 0
    # home_draws_ah = 0
    home_draws_total = 0
    # home_loses_ah = 0
    home_loses_total = 0
    # home_goals_ah = 0
    home_goals_total = 0
    # home_goals_conceded_ah = 0
    home_goals_conceded_total = 0
    away_matches = 0
    # away_wins_ah = 0
    away_wins_total = 0
    # away_draws_ah = 0
    away_draws_total = 0
    # away_loses_ah = 0
    away_loses_total = 0
    # away_goals_scored_ah = 0
    away_goals_total = 0
    # away_goals_conceded_ah = 0
    away_goals_conceded_total = 0

    if curr_match['FTR'] == 'H':
        home_wins_total = 1
        away_loses_total = 1
    elif curr_match['FTR'] == 'A':
        away_wins_total = 1
        home_loses_total = 1
    else:
        home_draws_total = 1
        away_draws_total = 1

    values_to_add = [1, home_wins_total, home_loses_total, home_draws_total, curr_match['FTHG'], curr_match['FTAG'],
                     1, away_wins_total, away_loses_total, away_draws_total, curr_match['FTAG'], curr_match['FTHG']]

    for key in team[curr_match['HomeTeam']]:
        print(team[key])

    for key in team[curr_match['HomeTeam']].keys():
        list_of_matches = team[key]
        for match in list_of_matches:
            print(match)

    if curr_match['HomeTeam'] in team:
        for key in team[curr_match['HomeTeam']].keys:
            list_of_matches = team[key]
            for match in list_of_matches:
                match[key].second = [sum(x) for x in zip(match[match_index].second, values_to_add)]
    elif curr_match['AwayTeam'] in team:
        for key in team[curr_match['HomeTeam']].values():
            match[key].second = [sum(x) for x in zip(match[match_index].second, values_to_add)]
    else:
        match[match_index] = ((curr_match['HomeTeam'], curr_match['AwayTeam']), [])

    if curr_match['HomeTeam'] in team:
        team[curr_match['HomeTeam']] = team[curr_match['HomeTeam']] + [match_index]
    else:
        team[curr_match['HomeTeam']] = [match_index]
    if curr_match['AwayTeam'] in team:
        team[curr_match['AwayTeam']] = team[curr_match['AwayTeam']] + [match_index]
    else:
        team[curr_match['AwayTeam']] = [match_index]

    for key, value in match.items():
        print(key, value)

    for key, value in team.items():
        print(key, value)

    print("####################################################################")

    # print(match)
    # print(team)


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

