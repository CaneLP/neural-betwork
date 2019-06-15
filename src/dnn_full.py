import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import adam, sgd, rmsprop
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# Beautify print - delete later
import sys
np.set_printoptions(threshold=sys.maxsize)


def data():
    matches_nn_input = []
    output_final_ints = []
    with open('processed_data_full.json') as json_file:
        json_data = json.load(json_file)
        for (key, value) in json_data.items():
            match_nn_input = []
            match_nn_input.append(value[0]['home_team_wins'])
            match_nn_input.append(value[0]['home_team_draws'])
            match_nn_input.append(value[0]['home_team_losses'])
            match_nn_input.append(value[0]['home_team_goals_scored'])
            match_nn_input.append(value[0]['home_team_goals_conceded'])
            match_nn_input.append(value[0]['home_team_shots'])
            match_nn_input.append(value[0]['home_team_shots_on_target'])
            match_nn_input.append(value[0]['home_team_shots_opposition'])
            match_nn_input.append(value[0]['home_team_shots_opposition_on_target'])
            match_nn_input.append(value[0]['home_team'])
            match_nn_input.append(value[0]['away_team_wins'])
            match_nn_input.append(value[0]['away_team_draws'])
            match_nn_input.append(value[0]['away_team_losses'])
            match_nn_input.append(value[0]['away_team_goals_scored'])
            match_nn_input.append(value[0]['away_team_goals_conceded'])
            match_nn_input.append(value[0]['away_team_shots'])
            match_nn_input.append(value[0]['away_team_shots_on_target'])
            match_nn_input.append(value[0]['away_team_shots_opposition'])
            match_nn_input.append(value[0]['away_team_shots_opposition_on_target'])
            match_nn_input.append(value[0]['away_team'])
            match_nn_input.append(value[0]['home_team_bet'])
            match_nn_input.append(value[0]['draw_bet'])
            match_nn_input.append(value[0]['away_team_bet'])
            matches_nn_input.append(match_nn_input)
            output_final_ints.append(value[0]['result'])

    matches_stats = np.array(matches_nn_input)
    output_final_ints = np.array(output_final_ints)

    teams_strengths = {}
    with open('processed_teams_stats.json') as json_file:
        json_data = json.load(json_file)
        for (key, value) in json_data.items():
            home_strength = ((value[0]['home_wins'] + value[0]['home_draws'] / 2.0 - value[0]['home_losses']) /
                             value[0]['num_matches']) + \
                            ((value[0]['home_goals_scored'] - (value[0]['home_goals_conceded'])) /
                             value[0]['num_matches'])
            away_strength = ((value[0]['away_wins'] + value[0]['away_draws'] / 2.0 - value[0]['away_losses']) /
                             value[0]['num_matches']) + \
                            ((value[0]['away_goals_scored'] - (value[0]['away_goals_conceded'])) /
                             value[0]['num_matches'])
            teams_strengths[key] = [home_strength, away_strength]

    # print(teams_strengths)

    htw = 0
    htd = 1
    htl = 2
    htgs = 3
    htgc = 4
    hts = 5
    htsot = 6
    htso = 7
    htsoot = 8
    ht = 9
    atw = 10
    atd = 11
    atl = 12
    atgs = 13
    atgc = 14
    ats = 15
    atsot = 16
    atso = 17
    atsoot = 18
    at = 19
    htb = 20
    db = 21
    atb = 22

    coefficient_win_rate = 10
    coefficient_goals = 5
    coefficient_shots = 0
    coefficient_bet = 0

    nn_input = []
    for stats in matches_stats:
        # print(teams_strengths[stats[ht]])
        calculate_input_ht = teams_strengths[stats[ht]][0] * \
                             (coefficient_win_rate * (float(stats[htw]) + float(stats[htd]) / 2.0 -
                                                      float(stats[htl])) +
                              coefficient_goals * (float(stats[htgs]) / (float(stats[htgs]) + float(stats[htgc]) + 1)) +
                              coefficient_shots * (float(stats[hts]) / float(stats[htsot]) -
                                                   float(stats[htso]) / float(stats[htsoot])) +
                              coefficient_bet * ((float(stats[htb]) + float(stats[db]) / 2.0) /
                                                 (float(stats[htb]) + float(stats[db]) + float(stats[atb]))))

        calculate_input_at = teams_strengths[stats[at]][1] *\
                             (coefficient_win_rate * (float(stats[atw]) + float(stats[atd]) / 2.0 -
                                                             float(stats[atl])) +
                              coefficient_goals * (float(stats[atgs]) / (float(stats[atgs]) + float(stats[atgc]) + 1)) +
                              coefficient_shots * (float(stats[ats]) / float(stats[atsot]) -
                                                   float(stats[atso]) / float(stats[atsoot])) +
                              coefficient_bet * ((float(stats[atb]) + float(stats[db]) / 2.0) /
                                                 (float(stats[htb]) + float(stats[db]) + float(stats[atb]))))

        nn_input.append([calculate_input_ht, calculate_input_at])

    print(nn_input)

    nn_input = np.array(nn_input)
    scaler = MinMaxScaler(feature_range=(0, 1))
    nn_input = scaler.fit_transform(nn_input)

    train_input, test_input, train_output, test_output = \
        train_test_split(nn_input, output_final_ints, test_size=0.3, shuffle=False)

    # Normalized input
    # max_col_values_train = [max(l) for l in list(zip(*train_input))]
    # # print("max column training values:", max_col_values_train)
    #
    # train_input = [list(zip(line, max_col_values_train)) for line in train_input]
    # train_input = [[t[0] / t[1] for t in line] for line in train_input]
    # train_input = np.array(train_input)
    # # print(train_input)
    #
    # max_col_values_test = [max(l) for l in list(zip(*test_input))]
    # # print("max column test values:", max_col_values_test)
    # test_input = [list(zip(line, max_col_values_test)) for line in test_input]
    # test_input = [[t[0] / t[1] for t in line] for line in test_input]
    # test_input = np.array(test_input)
    # print(test_input)

    return train_input, train_output, test_input, test_output, matches_nn_input


def create_model(train_input, train_output, test_input, test_output):

    output_class = ['H', 'D', 'A']

    model = Sequential()

    # Input layer and first hidden layer
    model.add(Dense({{choice([10, 20, 30, 40])}}, input_shape=(train_input.shape[1], )))
    # model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])}}))

    # Second hidden layer
    model.add(Dense({{choice([10, 20, 30, 40])}}))
    # model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])}}))

    # Output layer
    model.add(Dense(len(output_class)))
    model.add(Activation('softmax'))

    adam_opt = adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd_opt = sgd(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    optimizers = {{choice(['adam', 'sgd'])}}

    if optimizers == 'adam':
        optim = adam_opt
    else:
        optim = sgd_opt

    model.compile(optimizer=optim,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    result = model.fit(train_input, train_output, epochs={{choice([1, 2, 3, 4, 5])}}, batch_size=1)

    # print("Testing...")
    # test_loss, test_acc = model.evaluate(test_input, test_output)
    # print('Test accuracy:', test_acc)
    #
    # prediction = model.predict(test_input)
    #
    # print(result.history)

    validation_acc = np.amax(result.history['acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {
        'loss': -validation_acc,
        'status': STATUS_OK,
        'model': model
    }


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          eval_space=True)

    train_input, train_output, test_input, test_output, matches_nn_input = data()

    ######## Used for testing and exploring, delete later ############
    # output_class = ['H', 'D', 'A']
    #
    # model = Sequential()
    # model.add(Dense(30, input_shape=(train_input.shape[1],)))
    # model.add(Dropout(0.6))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(20))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(len(output_class)))
    # model.add(Activation('softmax'))
    #
    # adam = adam(lr=0.001)
    # model.compile(optimizer=adam,
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # result = model.fit(train_input, train_output, epochs=4, batch_size=1)
    #
    # print("Testing...")
    # test_loss, test_acc = model.evaluate(test_input, test_output)
    # print('Test accuracy:', test_acc)
    #
    # prediction = model.predict(test_input)

    ######################################################

    # TODO maybe: plot model only if user wants that
    # plot_model(best_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # print("---")
    # print(model.evaluate(test_input, test_output))

    print("best run:")
    # print("Activation function: ", activation_function_choice[best_run['Activation']])
    print(best_run)

    prediction = best_model.predict(test_input)

    ones = 0
    zeros = 0
    twos = 0
    correct_zeros = 0
    correct_twos = 0
    correct_ones = 0
    # print("PREDICTION")
    # print(prediction)
    # print("PREDICTION")
    for i in range(len(prediction)):
        # print(test_output[i], np.argmax(prediction[i]))
        if np.argmax(prediction[i]) == 0:
            zeros += 1
            if test_output[i] == 0:
                correct_zeros += 1
        elif np.argmax(prediction[i]) == 1:
            if test_output[i] == 1:
                correct_ones += 1
            ones += 1
        else:
            if test_output[i] == 2:
                correct_twos += 1
            twos += 1

    print()
    print(ones, zeros, twos)
    print()
    print(correct_ones, correct_zeros, correct_twos)

    # Confusion matrix
    pred_classes = best_model.predict_classes(test_input)
    # print("Predicted classes:")
    # print(np.count_nonzero(pred_classes == 1), np.count_nonzero(pred_classes == 0), np.count_nonzero(pred_classes == 2))

    cm = confusion_matrix(test_output, pred_classes)
    cm_plot_labels = ["Draw", "Home", "Away"]
    plot_confusion_matrix(cm, cm_plot_labels, title="Confusion matrix")


