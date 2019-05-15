import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import adam, sgd, rmsprop
from keras.utils.vis_utils import plot_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.model_selection import train_test_split
import json

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
            match_nn_input.append(value[0]['away_team_wins'])
            match_nn_input.append(value[0]['away_team_draws'])
            match_nn_input.append(value[0]['away_team_losses'])
            match_nn_input.append(value[0]['away_team_goals_scored'])
            match_nn_input.append(value[0]['away_team_goals_conceded'])
            match_nn_input.append(value[0]['away_team_shots'])
            match_nn_input.append(value[0]['away_team_shots_on_target'])
            match_nn_input.append(value[0]['away_team_shots_opposition'])
            match_nn_input.append(value[0]['away_team_shots_opposition_on_target'])
            # match_nn_input.append(value[0]['home_team_bet'])
            # match_nn_input.append(value[0]['draw_bet'])
            # match_nn_input.append(value[0]['away_team_bet'])
            matches_nn_input.append(match_nn_input)
            output_final_ints.append(value[0]['result'])

    matches_nn_input = np.array(matches_nn_input)
    output_final_ints = np.array(output_final_ints)

    train_input, test_input, train_output, test_output = \
        train_test_split(matches_nn_input, output_final_ints, test_size=0.5, shuffle=False)

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

    # choices = [10, 20, 40, 80, 160]
    # choices2 = [128, 256, 512, 1024]

    model = Sequential()
    model.add(Dense({{choice([10, 20, 40, 80, 160])}}, input_shape=(train_input.shape[1], )))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([10, 20, 40, 80, 160])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([10, 20, 40, 80, 160])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
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

    # print(result.history)
    validation_acc = np.amax(result.history['acc'])
    return {
        'loss': -validation_acc,
        'status': STATUS_OK,
        'model': model
    }


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          eval_space=True)

    train_input, train_output, test_input, test_output, matches_nn_input = data()

    # TODO maybe: plot model only if user wants that
    # plot_model(best_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print("---")
    print(best_model.evaluate(test_input, test_output))

    print("best run:")
    # print("Activation function: ", activation_function_choice[best_run['Activation']])
    print(best_run)

    # prediction = best_model.predict(test_input)
    #
    # print(np.count_nonzero(test_output == 0))
    # print(np.count_nonzero(test_output == 1))
    # print(np.count_nonzero(test_output == 2))
    #
    # ones = 0
    # zeros = 0
    # twos = 0
    # correct_zeros = 0
    # correct_twos = 0
    # correct_ones = 0
    # print("PREDICTION")
    # print(prediction)
    # print("PREDICTION")
    # for i in range(1347):
    #     #     print(test_output[i], np.argmax(prediction[i]))
    #     if np.argmax(prediction[i]) == 0:
    #         zeros += 1
    #         if test_output[i] == 0:
    #             correct_zeros += 1
    #     elif np.argmax(prediction[i]) == 1:
    #         if test_output[i] == 1:
    #             correct_ones += 1
    #         ones += 1
    #     else:
    #         if test_output[i] == 2:
    #             correct_twos += 1
    #         twos += 1
    #
    # print()
    # print(ones, zeros, twos)
    # print()
    # print(correct_ones, correct_zeros, correct_twos)

