## World Cup 2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Sbopen,Pitch, VerticalPitch, FontManager
import math

parser = Sbopen()
df_match = parser.match(competition_id=43, season_id=106)

df_matches = {}
for i, id in enumerate(df_match['match_id']):
    df_matches[id] = {}
    df_matches[id]['event'], df_matches[id]['related'], df_matches[id]['freeze'], df_matches[id]['tactic'] = parser.event(id)
df_frame, df_visible = parser.frame(3869685)
#first game of WC 2022

# exploring the data
def draw_position(sequence_id):
  sequence=df_frame[df_frame['id']==sequence_id]

  actor=sequence[sequence['actor']==True]
  actor_position = [(actor['x'].iloc[0], actor['y'].iloc[0])]

  teammate = sequence[(sequence['teammate'] == True) & (sequence['actor'] == False)]
  teammate_position=[(teammate['x'].iloc[i], teammate['y'].iloc[i]) for i in range(len(teammate))]

  opponent=sequence[sequence['teammate']==False]
  opponent_position=[(opponent['x'].iloc[i], opponent['y'].iloc[i]) for i in range(len(opponent))]

  return draw_grid_on_pitch(actor_position, teammate_position, opponent_position)

def calculate_xag(row):


  sequence=df_frame[df_frame['id']==row['id']]
  teammate = sequence[(sequence['teammate'] == True) & (sequence['actor'] == False)]
  teammate_position=[(teammate['x'].iloc[i], teammate['y'].iloc[i]) for i in range(len(teammate))]

  xag_product =1.0
  for position in teammate_position:
    modified_row = row.copy()
    modified_row['x'], modified_row['y'] = position

    # 각 팀메이트의 위치에서 xG 계산
    xg_value = calculate_xg_adv(modified_row)

    # 득점하지 못할 확률을 누적 곱함
    xag_product *= (1 - xg_value)

    # 최종 xAG 값 계산
  xag_result = 1 - xag_product
  return xag_result

df_evaluate = df_matches[3869685]['event'].copy()  # .copy()를 사용하여 명확하게 복사본을 만들어 작업
evaluate_mask = (df_evaluate['type_name'].isin(['Clearance', 'Pass', 'Shot', 'Carry'])) & (df_evaluate['period'] <= 4)
df_evaluate = df_evaluate[evaluate_mask]

df_evaluate['xg'] = df_evaluate.apply(lambda row: calculate_xg_adv(row), axis=1)

df_summary=df_evaluate[['id','timestamp','possession_team_name','type_name','x','y','end_x','end_y','pass_height_name','xg', 'outcome_name']].copy()
df_summary['xg'] = df_summary['xg'].round(4)


