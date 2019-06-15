import pandas as pd
import numpy as np
import copy
import glob
import json


def process_data():
    path = '../data/full'

    files = [f for f in glob.glob(path + "/*.csv", recursive=True)]

    matches = pd.DataFrame()

    for f in files:
        fields = ['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'WHH', 'WHD', 'WHA']
        curr_season = pd.read_csv(f, error_bad_lines=False, usecols=fields)
        curr_season.dropna(inplace=True)
        curr_season['FTHG'] = curr_season['FTHG'].astype(int)
        curr_season['FTAG'] = curr_season['FTAG'].astype(int)
        matches = matches.append(curr_season, ignore_index=True, sort='False')

    last_n_games = 10

    match = {}
    team = {}
    team_stats = {}
    for i in range(len(matches)):
        match_index = len(matches) - i - 1
        curr_match = matches.iloc[len(matches) - i - 1]
        match[match_index] = [(curr_match['HomeTeam'], curr_match['AwayTeam']), [0] * 11, [0] * 11,
                              [curr_match['WHH'], curr_match['WHD'], curr_match['WHA']]
                              ]

        if curr_match['HomeTeam'] in team:
            team[curr_match['HomeTeam']] = team[curr_match['HomeTeam']] + [match_index]
        else:
            team[curr_match['HomeTeam']] = [match_index]

        if curr_match['AwayTeam'] in team:
            team[curr_match['AwayTeam']] = team[curr_match['AwayTeam']] + [match_index]
        else:
            team[curr_match['AwayTeam']] = [match_index]

        if curr_match['HomeTeam'] in team_stats:
            win = 0
            draw = 0
            loss = 0
            num_matches = 1
            if curr_match['FTR'] == 'H':
                win = 1
            elif curr_match['FTR'] == 'D':
                draw = 1
            else:
                loss = 1
            goals_scored = curr_match['FTHG']
            goals_conceded = curr_match['FTAG']
            team_stats[curr_match['HomeTeam']][0] = [sum(x) for x in zip(team_stats[curr_match['HomeTeam']][0],
                                                                         [win, draw, loss, goals_scored, goals_conceded,
                                                                          num_matches])]
        else:
            win = 0
            draw = 0
            loss = 0
            if curr_match['FTR'] == 'H':
                win = 1
            elif curr_match['FTR'] == 'D':
                draw = 1
            else:
                loss = 1
            team_stats[curr_match['HomeTeam']] = [[win, draw, loss, curr_match['FTHG'], curr_match['FTAG'], 1], [0] * 6]

        if curr_match['AwayTeam'] in team_stats:
            win = 0
            draw = 0
            loss = 0
            num_matches = 1
            if curr_match['FTR'] == 'A':
                win = 1
            elif curr_match['FTR'] == 'D':
                draw = 1
            else:
                loss = 1
            goals_scored = curr_match['FTAG']
            goals_conceded = curr_match['FTHG']
            team_stats[curr_match['AwayTeam']][1] = [sum(x) for x in zip(team_stats[curr_match['AwayTeam']][1],
                                                                         [win, draw, loss, goals_scored, goals_conceded,
                                                                          num_matches])]
        else:
            win = 0
            draw = 0
            loss = 0
            if curr_match['FTR'] == 'A':
                win = 1
            elif curr_match['FTR'] == 'D':
                draw = 1
            else:
                loss = 1
            team_stats[curr_match['AwayTeam']] = [[0] * 6, [win, draw, loss, curr_match['FTAG'], curr_match['FTHG'], 1]]

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

            shots_total = 0
            shots_on_target_total = 0
            shots_op = 0
            shots_on_target_op = 0

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

                    shots_total += match_calc['HS']
                    shots_on_target_total += match_calc['HST']
                    shots_op += match_calc['AS']
                    shots_on_target_op += match_calc['AST']
                elif match_calc['AwayTeam'] == curr_team:
                    goals_scored_total += match_calc['FTAG']
                    goals_conceded_total += match_calc['FTHG']

                    shots_total += match_calc['AS']
                    shots_on_target_total += match_calc['AST']
                    shots_op += match_calc['HS']
                    shots_on_target_op += match_calc['HST']

            home_or_away = 2
            if matches.iloc[match_key]['HomeTeam'] == curr_team:
                home_or_away = 1
            match[match_key][home_or_away] = [games_total, wins_total, draws_total,
                                              losses_total, goals_scored_total, goals_conceded_total,
                                              shots_total, shots_on_target_total, shots_op, shots_on_target_op,
                                              curr_team]

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

            shots_total = 0
            shots_on_target_total = 0
            shots_op = 0
            shots_on_target_op = 0

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

                    shots_total += match_calc['HS']
                    shots_on_target_total += match_calc['HST']
                    shots_op += match_calc['AS']
                    shots_on_target_op += match_calc['AST']
                elif match_calc['AwayTeam'] == curr_team:
                    goals_scored_total += match_calc['FTAG']
                    goals_conceded_total += match_calc['FTHG']

                    shots_total += match_calc['AS']
                    shots_on_target_total += match_calc['AST']
                    shots_op += match_calc['HS']
                    shots_on_target_op += match_calc['HST']

            home_or_away = 2
            if matches.iloc[match_key]['HomeTeam'] == curr_team:
                home_or_away = 1
            match[match_key][home_or_away] = [games_total, wins_total, draws_total,
                                              losses_total, goals_scored_total, goals_conceded_total,
                                              shots_total, shots_on_target_total, shots_op, shots_on_target_op,
                                              curr_team]

    # Create an array with the data for the neural network input
    matches_nn_input = []
    rows_to_drop = []
    for key, value in match.items():
        if np.count_nonzero(match[key][1]) == 0 or np.count_nonzero(match[key][2]) == 0:
            rows_to_drop.append(key)
        else:
            matches_nn_input.append(match[key][1][1:] + match[key][2][1:] + match[key][3])

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
            'home_team': matches_nn_input[i][9],
            'home_team_wins': float(matches_nn_input[i][0]),
            'home_team_draws': float(matches_nn_input[i][1]),
            'home_team_losses': float(matches_nn_input[i][2]),
            'home_team_goals_scored': float(matches_nn_input[i][3]),
            'home_team_goals_conceded': float(matches_nn_input[i][4]),
            'home_team_shots': float(matches_nn_input[i][5]),
            'home_team_shots_on_target': float(matches_nn_input[i][6]),
            'home_team_shots_opposition': float(matches_nn_input[i][7]),
            'home_team_shots_opposition_on_target': float(matches_nn_input[i][8]),
            'away_team': matches_nn_input[i][19],
            'away_team_wins': float(matches_nn_input[i][10]),
            'away_team_draws': float(matches_nn_input[i][11]),
            'away_team_losses': float(matches_nn_input[i][12]),
            'away_team_goals_scored': float(matches_nn_input[i][13]),
            'away_team_goals_conceded': float(matches_nn_input[i][14]),
            'away_team_shots': float(matches_nn_input[i][15]),
            'away_team_shots_on_target': float(matches_nn_input[i][16]),
            'away_team_shots_opposition': float(matches_nn_input[i][17]),
            'away_team_shots_opposition_on_target': float(matches_nn_input[i][18]),
            'home_team_bet': float(matches_nn_input[i][20]),
            'draw_bet': float(matches_nn_input[i][21]),
            'away_team_bet': float(matches_nn_input[i][22]),
            'result': int(output_final_ints[i])
        })

    with open('processed_data.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    json_data = {}
    for key, value in team_stats.items():
        json_data[key] = []
        json_data[key].append({
            'home_wins': int(value[0][0]),
            'home_draws': int(value[0][1]),
            'home_losses': int(value[0][2]),
            'home_goals_scored': int(value[0][3]),
            'home_goals_conceded': int(value[0][4]),
            'away_wins': int(value[1][0]),
            'away_draws': int(value[1][1]),
            'away_losses': int(value[1][2]),
            'away_goals_scored': int(value[1][3]),
            'away_goals_conceded': int(value[1][4]),
            'num_matches': int(value[0][5] + value[1][5])
        })

    with open('processed_teams_stats.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


if __name__ == '__main__':
    process_data()
