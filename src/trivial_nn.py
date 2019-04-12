import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import activations


season_1 = pd.read_csv('../data/season_16-17.csv')
season_2 = pd.read_csv('../data/season_17-18.csv')
season_3 = pd.read_csv('../data/season_18-19.csv')

matches = season_1.append(season_2, ignore_index=True)
matches = matches.append(season_3, sort='False', ignore_index=True)
# print(matches)

# exit()

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

