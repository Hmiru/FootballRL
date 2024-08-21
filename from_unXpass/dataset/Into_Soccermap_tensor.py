import pandas as pd
import torch
from from_unXpass.dataset.utils import *
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from functools import wraps
import numpy as np
import socceraction.spadl.config as spadlconfig
from typing import List

def intended(actionfn):
    @wraps(actionfn)
    def _wrapper(self, sample):
        actions = pd.DataFrame([sample])

        failed_passes = actions[actions["success"] == 0]
        for idx, action in failed_passes.iterrows():
            if action["freeze_frame_360"] is None:
                continue

            receiver_coo = np.array(
                [
                    (o["x"], o["y"])
                    for o in action["freeze_frame_360"]
                    if o["teammate"] and not o["actor"]
                ]
            )

            if len(receiver_coo) == 0:
                continue
            ball_coo = np.array([action.start_x, action.start_y])
            interception_coo = np.array([action.end_x, action.end_y])
            dist = np.sqrt(
                (receiver_coo[:, 0] - interception_coo[0]) ** 2
                + (receiver_coo[:, 1] - interception_coo[1]) ** 2
            )
            a = interception_coo - ball_coo
            b = receiver_coo - ball_coo
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a > 0 and norm_b > 0:
                angle = np.arccos(
                    np.clip(
                        np.sum(a * b, axis=1) / (norm_a * norm_b), -1, 1
                    )
                )
                if np.amin(angle) > 0.35:
                    continue
                too_wide = np.where(angle > 0.35)[0]
                dist[too_wide] = np.inf
                exp_receiver = np.argmax((np.amin(dist) / dist) * (np.amin(angle) / angle))
                actions.loc[idx, ["end_x", "end_y"]] = receiver_coo[exp_receiver]

        sample = actions.iloc[0].to_dict()
        return actionfn(self, sample)
    return _wrapper


class ToSoccerMapTensor:
    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def align_left_to_right(self, sample):
        if not sample['if_home_team']:
            sample["start_x"] = spadlconfig.field_length - sample["start_x"]
            sample["end_x"] = spadlconfig.field_length - sample["end_x"]
            sample["start_y"] = spadlconfig.field_width - sample["start_y"]
            sample["end_y"] = spadlconfig.field_width - sample["end_y"]

            # Adjust the freeze frame coordinates for the away team
            for player in sample["freeze_frame_360"]:
                player["x"] = spadlconfig.field_length - player["x"]
                player["y"] = spadlconfig.field_width - player["y"]

        return sample


    @intended
    def __call__(self, sample):



        sample = self.align_left_to_right(sample)
        start_x, start_y, end_x, end_y = (
            sample["start_x"],
            sample["start_y"],
            sample["end_x"],
            sample["end_y"],
        )

        frame = pd.DataFrame.from_records(sample["freeze_frame_360"])
        target = int(sample["success"]) if "success" in sample else None

        ball_coo = np.array([[start_x, start_y]])
        goal_coo = np.array([[105, 34]])
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(
            -1, 2
        )

        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        matrix = np.zeros((7, self.y_bins, self.x_bins))

        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        yy, xx = np.ogrid[0.5: self.y_bins, 0.5: self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)

        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1


        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            None,
        )

def convert_row_to_sample(row):
    '''
    주어진 행 변환하여 샘플 형식으로 만듭니다.

    :param row: pandas DataFrame의 한 행을 나타냅니다. 이 행에는 시작 위치, 끝 위치, 팀원 및 상대 선수들의 위치 정보 등이 포함되어 있습니다.
    :return: 모델 학습에 사용할 수 있는 딕셔너리 형식의 샘플을 반환합니다.
    '''
    freeze_frame = []
    freeze_frame.append({'x': row['x'], 'y': row['y'], 'teammate': True, 'actor': True})

    for i in range(1, 11):
        if pd.notnull(row[f'teammate_{i}_x']) and pd.notnull(row[f'teammate_{i}_y']):
            freeze_frame.append({'x': row[f'teammate_{i}_x'], 'y': row[f'teammate_{i}_y'], 'teammate': True, 'actor': False})
    for i in range(1, 11):
        if pd.notnull(row[f'opponent_{i}_x']) and pd.notnull(row[f'opponent_{i}_y']):
            freeze_frame.append({'x': row[f'opponent_{i}_x'], 'y': row[f'opponent_{i}_y'], 'teammate': False, 'actor': False})
    sample = {
        'start_x': row['x'],
        'start_y': row['y'],
        'end_x': row['end_x'],
        'end_y': row['end_y'],
        'freeze_frame_360': freeze_frame,
        'success': 1 if row['outcome_name'] == 'success' else 0,
        'if_home_team': row['if_home_team']
    }
    return sample


if __name__ == "__main__":
    final_df = pd.read_csv("WC_EU_LEV_data.csv", index_col=0)
    row_data = final_df.iloc[8]
    sample = convert_row_to_sample(row_data)
    #
    # print("초기 샘플 데이터:")
    # print(sample)
    # print(f"Success 상태: {sample['success']}")
    # print(f"Freeze Frame 데이터: {sample['freeze_frame_360']}")

    tensor_converter = ToSoccerMapTensor()
    matrix, mask, target = tensor_converter(sample)

    # print("Matrix 크기:", matrix.shape)  # (7, 68, 104)
    print("Mask 크기:", mask.shape)  # (1, 68, 104)
    print("Target 값:", target)  # tensor([1.])

