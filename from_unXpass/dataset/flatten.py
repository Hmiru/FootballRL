import pandas as pd

def flatten_columns(row, df_frame):
    sequence = df_frame[df_frame['id'] == row['id']]

    actor = sequence[sequence['actor'] == True]
    actor_position = [(actor['x'].iloc[0], actor['y'].iloc[0])] if not actor.empty else [(None, None)]

    keeper = sequence[(sequence['keeper'] == True) & (sequence['teammate'] == False)]
    keeper_position = [(keeper['x'].iloc[0], keeper['y'].iloc[0])] if not keeper.empty else [(None, None)]

    teammate = sequence[(sequence['teammate'] == True) & (sequence['actor'] == False) | (sequence['keeper'] == True) & (sequence['teammate'] == True)]
    teammate_positions = [(teammate['x'].iloc[i], teammate['y'].iloc[i]) for i in range(len(teammate))]

    opponent = sequence[(sequence['teammate'] == False) & (sequence['keeper'] == False)]
    opponent_positions = [(opponent['x'].iloc[i], opponent['y'].iloc[i]) for i in range(len(opponent))]

    # 새로운 Dictionary 초기화
    flattened = {
        'id': row['id'],
        'actor_x': actor_position[0][0],
        'actor_y': actor_position[0][1],
        'keeper_x': keeper_position[0][0],
        'keeper_y': keeper_position[0][1],
        'x': row['x'],
        'y': row['y'],
        'end_x': row['end_x'],
        'end_y': row['end_y'],
        'outcome_name': row['outcome_name']
    }

    # 각 teammate 위치를 별도의 열로 추가
    for i in range(10):
        if i < len(teammate_positions):
            flattened[f'teammate_{i+1}_x'] = teammate_positions[i][0]
            flattened[f'teammate_{i+1}_y'] = teammate_positions[i][1]
        else:
            flattened[f'teammate_{i+1}_x'] = None
            flattened[f'teammate_{i+1}_y'] = None

    # 각 opponent 위치를 별도의 열로 추가
    for i in range(10):
        if i < len(opponent_positions):
            flattened[f'opponent_{i+1}_x'] = opponent_positions[i][0]
            flattened[f'opponent_{i+1}_y'] = opponent_positions[i][1]
        else:
            flattened[f'opponent_{i+1}_x'] = None
            flattened[f'opponent_{i+1}_y'] = None

    return pd.Series(flattened)