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

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import adam, sgd, rmsprop

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


np.set_printoptions(threshold=sys.maxsize)


def data():
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

    return train_input, train_output, test_input, test_output, matches_nn_input


def create_model(train_input, train_output, test_input, test_output, matches_nn_input):

    output_class = ['H', 'D', 'A']

    model = Sequential()
    model.add(Dense({{choice([128, 256, 512, 1024])}}, input_shape=(matches_nn_input.shape[1], )))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    # model.add(Dense({{choice([128, 256, 512, 1024])}}))
    # model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense(len(output_class)))
    model.add(Activation('softmax'))

    adam_opt = adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd_opt = sgd(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    optimizers = {{choice(['sgd', 'adam'])}}

    if optimizers == 'adam':
        optim = adam_opt
    else:
        optim = sgd_opt

    model.compile(optimizer=optim,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    result = model.fit(train_input, train_output, epochs={{choice([8, 9, 10, 11, 12, 13])}})

    # print("Testing...")
    # test_loss, test_acc = model.evaluate(test_input, test_output)
    # print('Test accuracy:', test_acc)

    # prediction = model.predict(test_input)

    print(result.history)
    validation_acc = np.amax(result.history['acc'])
    return {
        'loss': -validation_acc,
        'status': STATUS_OK,
        'model': model
    }

# for i in range(21):
#     print(test_output[i])
#     print(np.argmax(prediction[i]))
#     print("-----------------------")


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    train_input, train_output, test_input, test_output, matches_nn_input = data()
    print("---")
    print(best_model.evaluate(test_input, test_output))
    print("best run:")
    print(best_run)

