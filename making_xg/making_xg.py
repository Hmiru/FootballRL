import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Sbopen,Pitch, VerticalPitch, FontManager
import math

parser = Sbopen()
## World Cup 2022
##df_match = parser.match(competition_id=43, season_id=106)

## Euro 2020
## df_match = parser.match(competition_id=55, season_id=43)

## Serie A 2015/16
df_match = parser.match(competition_id=12, season_id=27)
df_matches = {}

for i, id in enumerate(df_match['match_id']):
    df_matches[id] = {}
    df_matches[id]['event'], df_matches[id]['related'], df_matches[id]['freeze'], df_matches[id]['tactic'] = parser.event(id)

df_shot=pd.DataFrame(columns=['x', 'y', 'outcome_name', 'shot_statsbomb_xg'])

for id in df_match['match_id']:
  mask_shot = (df_matches[id]['event'].type_name=='Shot') & (df_matches[id]['event'].period<=4) &(df_matches[id]['event'].sub_type_name=='Open Play')
  shots_temp=df_matches[id]['event'].loc[mask_shot, ['x', 'y', 'outcome_name', 'shot_statsbomb_xg']]
  df_shot = pd.concat([df_shot, shots_temp]).reset_index(drop=True)


def calculate_angle(x, y):
  # 44 and 36 is the location of each goal post
  g0 = [120, 44]
  p = [x, y]
  g1 = [120, 36]

  v0 = np.array(g0) - np.array(p)
  v1 = np.array(g1) - np.array(p)

  angle = math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
  return(abs(np.degrees(angle)))

def calculate_distance(x, y):
  x_dist=120-x
  y_dist = abs(y - 40)  # y 거리는 y 좌표와 골대 중심 y 좌표(40)의 절대값 차이로 계산
  return math.sqrt(x_dist**2 + y_dist**2)

# More Advanced xG Model with New Features
new_features=['x', 'y', 'outcome_name', 'sub_type_name', 'body_part_name','under_pressure', 'shot_first_time', 'technique_name', 'shot_statsbomb_xg','play_pattern_name']
df_shot = pd.DataFrame()
for id in df_match['match_id']:
  mask_shot=(df_matches[id]['event'].type_name=='Shot') & (df_matches[id]['event'].period<=4)
  shots_temp=df_matches[id]['event'].loc[mask_shot,new_features]
  df_shot=pd.concat([df_shot, shots_temp]).reset_index(drop=True)
df_shot['angle']=df_shot.apply(lambda row:calculate_angle(row['x'], row['y']), axis=1)
df_shot['distance']=df_shot.apply(lambda row:calculate_distance(row['x'], row['y']), axis=1)

df_shot['under_pressure']=df_shot['under_pressure'].fillna(0)
df_shot['under_pressure']=df_shot['under_pressure'].astype(int)

df_shot['shot_first_time']=df_shot['shot_first_time'].fillna(0)
df_shot['shot_first_time']=df_shot['shot_first_time'].astype(int)

df_shot=pd.get_dummies(df_shot,columns=['body_part_name'])
df_shot=pd.get_dummies(df_shot,columns=['technique_name'])
df_shot=pd.get_dummies(df_shot, columns=['sub_type_name'])
df_shot=pd.get_dummies(df_shot, columns=['play_pattern_name'])
df_shot['goal']=df_shot.apply(lambda row:1 if row['outcome_name']=='Goal' else 0, axis=1)
df_shot['body_part_name_Foot'] = df_shot['body_part_name_Left Foot'] + df_shot['body_part_name_Right Foot']

# 불필요한 열 제거: 'Left Foot', 'Right Foot' 열 제거
df_shot.drop(['body_part_name_Left Foot', 'body_part_name_Right Foot', 'body_part_name_Other'], axis=1, inplace=True)
