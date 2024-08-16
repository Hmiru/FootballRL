import torch
from network import SoccerMap
import pandas as pd
from dataset import SoccerDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 모델 로드
input_size = 11
model = SoccerMap(input_size)
model.load_state_dict(torch.load("final_model_30.pth"))
model.eval()

# 테스트 데이터 로드
test_data = pd.read_csv("../dataset/total_state_data_with_paths.csv", index_col=0)
test_data = test_data.iloc[:, [5, 6, 7, -1]]
test_dataset = SoccerDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 히트맵 시각화 함수 정의
def plot_success_surface(success_surface, title="Success Surface", our_team=None, opponents=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(success_surface, annot=False, cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.gca().invert_yaxis()

    if our_team is not None:
        for player in our_team:
            plt.plot(player[1], player[0], 'bo')  # blue circles for our team

    if opponents is not None:
        for player in opponents:
            plt.plot(player[1], player[0], 'ro')  # red circles for opponents

    plt.show()

# 테스트 및 시각화
with torch.no_grad():
    for i, (state_inputs, _,_ , _) in enumerate(test_dataloader):
        if i == 23:  # 특정 인덱스 선택
            state_inputs = state_inputs.to(torch.float32)
            success_predictions = model(state_inputs)

            # 성공 확률 히트맵 생성
            success_predictions = success_predictions.view(80, 120).cpu().numpy()

            # 플레이어 위치 시각화
            state_inputs_np = state_inputs.squeeze().numpy()
            our_team_positions = np.argwhere(state_inputs_np[9] > 0)
            opponent_positions = np.argwhere(state_inputs_np[10] > 0)

            plot_success_surface(success_predictions, title="Predicted Success Surface with Player Positions", our_team=our_team_positions, opponents=opponent_positions)

            break  # 첫 번째 샘플만 처리
