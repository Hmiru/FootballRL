import os
import numpy as np
import pandas as pd
from rowtostate import RowToState

data = pd.read_csv("../raw_dataset/total_state_data.csv", index_col=0)

print(data.head())

# # 데이터 정규화 함수
def normalize_data(data):
    for col in data.columns:
        if col not in ['id', 'match_id', 'type_name', 'outcome_name', 'end_y', 'end_x']:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

# # 데이터 정규화
pd.set_option('display.max_columns', None)
test_data = normalize_data(data.iloc[0:10,:])
print(test_data.head())
#
# # 전처리된 상태를 저장할 디렉토리 생성
# output_dir = "../dataset/new_preprocessed_states"
# os.makedirs(output_dir, exist_ok=True)
#
# state_paths = []
#
# for idx, row in data.iterrows():
#     my_select = RowToState(row)
#     (
#         my_select.distance("ball").distance("goal").angle("goalpost", "cosine").angle("goalpost", "sine")
#         .angle("goalpost", "angle").angle("ball", "cosine").angle("ball", "sine").angle("ball", "angle")
#         .player_coor(row, role="actor").player_coor(row, role="teammate").player_coor(row, role="opponent")
#     )
#     state = my_select.stack_features()
#
#     # 파일로 저장
#     state_path = os.path.join(output_dir, f"state_{idx}.npy")
#     np.save(state_path, state)
#     state_paths.append(state_path)
#
# # 데이터프레임에 전처리된 상태 파일 경로 추가
# data['state_path'] = state_paths
#
# # 수정된 데이터프레임을 CSV 파일로 저장
# data.to_csv("../dataset/total_state_data_with_paths.csv")
