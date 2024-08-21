
from utils import *
from flatten import flatten_columns

'''
This script generates a comprehensive CSV file using the StatsBomb API.
'''

parser = Sbopen()

def final_df(competition_id, season_id):
    # Get match data for the specified competition and season
    df_match = parser.match(competition_id=competition_id, season_id=season_id)
    all_matches_df = []

    # Iterate over each match in the season
    for match_id in df_match['match_id']:
        home_team_id = df_match[df_match['match_id'] == match_id]['home_team_id'].values[0]

        # Get event and frame data for the current match
        df_event = parser.event(match_id)
        df_frame, df_visible = parser.frame(match_id)

        # Filter event data based on specific criteria
        evaluate_mask = (
            (df_event[0]['type_name'].isin(['Pass'])) &
            (df_event[0]['period'] <= 4) &
            (df_event[0]['play_pattern_name'].isin(['Regular Play', 'From Counter'])) &
            (df_event[0]['body_part_name'].isin(['Right Foot', 'Left Foot']))
        )
        df_evaluate = df_event[0][evaluate_mask]

        # Summarize relevant columns and add home team information
        df_summary = df_evaluate[['id', 'index', 'x', 'y', 'end_x', 'end_y', 'outcome_name', 'team_id']].copy()
        df_summary['outcome_name'] = df_summary['outcome_name'].fillna('success')

        df_summary['home_team_id'] = home_team_id
        df_summary['if_home_team'] = df_summary['team_id'] == home_team_id

        # Drop the now unnecessary columns
        df_summary.drop(['team_id', 'home_team_id'], axis=1, inplace=True)



        # Flatten columns and prepare the match DataFrame
        result_list = []
        for _, row in df_summary.iterrows():
            mid_df = flatten_columns(row, df_frame)
            result_list.append(mid_df)

        match_df = pd.concat(result_list, axis=1).transpose()
        all_matches_df.append(match_df)

    # Combine all match DataFrames into one final DataFrame
    final_df = pd.concat(all_matches_df, ignore_index=True)
    return final_df

if __name__=="__main__":
    #final_df = final_df(competition_id=43, season_id=106) # World Cup 2022
    # final_df.to_csv("world_cup_total_data_with_state_label_mask.csv")

    final_df = final_df(competition_id=55, season_id=282)  # Euro 2024
    final_df.to_csv("Euro_total_data_with_state_label_mask.csv")

    # final_df = final_df(competition_id=9, season_id=281)  # Leverkusen 23-24
    # final_df.to_csv("Leverkusen_total_data_with_state_label_mask.csv")
    #
    #final_df = final_df(competition_id=9, season_id=281)  # Leverkusen 23-24

