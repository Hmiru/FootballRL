import pandas as pd
'''
event data의 한 행을 처리하여, 그 행에 포함된 데이터를 보다 평평한 형태로 변환하는 역할을 합니다. 
이 함수는 특정 조건에 따라 행위자(actor), 골키퍼(keeper), 팀원(teammate), 상대팀(opponent)의 위치 데이터를 추출하고, 이를 여러 개의 열(column)로 평평하게 정리합니다.

row: 함수가 처리할 event data의 한 행입니다.
df_frame: frame data로, event_id를 기준으로 필요한 데이터를 추출합니다
'''

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

