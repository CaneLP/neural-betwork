import pandas as pd
import numpy as np
import copy
import glob
import json


def process_data():
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

    # TODO
    # add more data maybe

    # Collecting data
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

    json_data = {}
    for i in range(matches_nn_input.shape[0]):
        json_data[i] = []
        json_data[i].append({
            'home_team_wins': int(matches_nn_input[i][0]),
            'home_team_draws': int(matches_nn_input[i][1]),
            'home_team_losses': int(matches_nn_input[i][2]),
            'home_team_goals_scored': int(matches_nn_input[i][3]),
            'home_team_goals_conceded': int(matches_nn_input[i][4]),
            'away_team_wins': int(matches_nn_input[i][5]),
            'away_team_draws': int(matches_nn_input[i][6]),
            'away_team_losses': int(matches_nn_input[i][7]),
            'away_team_goals_scored': int(matches_nn_input[i][8]),
            'away_team_goals_conceded': int(matches_nn_input[i][9]),
            'result': int(output_final_ints[i])
        })

    with open('processed_data.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


if __name__ == '__main__':
    process_data()
