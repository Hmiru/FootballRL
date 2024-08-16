from making_xg import *

X_cols = ['angle', 'distance','under_pressure', 'shot_first_time', 'body_part_name_Foot','body_part_name_Head',
          'technique_name_Backheel','technique_name_Diving Header', 'technique_name_Half Volley',
          'technique_name_Lob', 'technique_name_Normal','technique_name_Overhead Kick', 'technique_name_Volley',
          'sub_type_name_Corner', 'sub_type_name_Free Kick','sub_type_name_Open Play', 'sub_type_name_Penalty',
          'play_pattern_name_From Counter', 'play_pattern_name_From Free Kick','play_pattern_name_From Goal Kick',
          'play_pattern_name_From Keeper', 'play_pattern_name_From Kick Off', 'play_pattern_name_From Throw In',
          'play_pattern_name_Other', 'play_pattern_name_Regular Play']

X = df_shot[X_cols]
y = df_shot['goal']

import numpy as np
from sklearn.model_selection import train_test_split
feature_to_scale=X[['angle','distance']]
scaler = StandardScaler()
scaled_feature= scaler.fit_transform(feature_to_scale)
mean = scaler.mean_
std = scaler.scale_
scaled_feature_df=pd.DataFrame(scaled_feature, columns=['angle','distance'])
X_scaled=X.copy()
X_scaled.update(scaled_feature_df)
import joblib
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=118)

adv_model = LogisticRegression()
adv_model.fit(X_train, y_train)
y_pred = adv_model.predict_proba(X_test)[:, 1]
metrics.brier_score_loss(y_test, y_pred)
joblib.dump(adv_model, '/content/adv_model.joblib')
def calculate_xg_adv(row):
  # print(type(row['shot_first_time']))
  # print(type(row['under_pressure']))

  shot_first_time = 0 if np.isnan(row['shot_first_time']) or row['shot_first_time'] is False else 1
  under_pressure = 0 if np.isnan(row['under_pressure']) or row['under_pressure'] is False else 1

  angle = calculate_angle(row['x'], row['y'])
  distance = calculate_distance(row['x'], row['y'])

  scaled_angle = (angle - mean[0]) / std[0]
  scaled_distance = (distance - mean[1]) / std[1]


  body_part_name = {'Foot': 0, 'Head': 0}
  play_pattern_name = {'From Counter': 0, 'From Free Kick': 0,
                         'From Goal Kick': 0, 'From Keeper': 0,
                         'From Kick Off': 0, 'From Throw In': 0,
                         'From Other': 0, 'From Regular Play': 0}
  technique_name = {'Backheel': 0, 'Diving Header': 0,
                      'Half Volley': 0, 'Lob': 0,
                      'Normal': 0, 'Overhead Kick': 0,
                      'Volley': 0}
  sub_type_name = {'Corner': 0, 'Free Kick': 0,
                     'Open Play': 0, 'Penalty': 0}

  body_part_name['Head'] = 1 if row['body_part_name'] == 'Head' else 0
  body_part_name['Foot'] = 1 if row['body_part_name'] in ['Left Foot', 'Right Foot'] else 0

  if row['play_pattern_name'] in play_pattern_name:
        play_pattern_name[row['play_pattern_name']] = 1
  if row['technique_name'] in technique_name:
        technique_name[row['technique_name']] = 1
  if row['sub_type_name'] in sub_type_name:
        sub_type_name[row['sub_type_name']] = 1




  X = [[scaled_angle, scaled_distance, under_pressure,  shot_first_time,

        body_part_name['Foot'], body_part_name['Head'],

        technique_name['Backheel'], technique_name['Diving Header'],
        technique_name['Half Volley'], technique_name['Lob'],
        technique_name['Normal'], technique_name['Overhead Kick'],technique_name['Volley'],

        sub_type_name['Corner'],sub_type_name['Free Kick'], sub_type_name['Open Play'],sub_type_name['Penalty'],

        play_pattern_name['From Counter'], play_pattern_name['From Free Kick'],
        play_pattern_name['From Goal Kick'], play_pattern_name['From Keeper'],
        play_pattern_name['From Kick Off'], play_pattern_name['From Throw In'],
        play_pattern_name['From Other'], play_pattern_name['From Regular Play']]]


  xg = adv_model.predict_proba(X)[:, 1][0]
  return xg