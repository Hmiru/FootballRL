from from_unXpass.dataset.utils import *


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        start_x, start_y, end_x, end_y = (
            sample["start_x"],
            sample["start_y"],
            sample["end_x"],
            sample["end_y"],
        )
        frame = pd.DataFrame.from_records(sample["freeze_frame_360"])
        target = int(sample["success"]) if "success" in sample else None

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])
        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(
            -1, 2
        )
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # Output
        matrix = np.zeros((7, self.y_bins, self.x_bins))

        # CH 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        # CH 3: Distance to ball
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # Mask
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
        'speedx': 0,  # 속도 데이터가 없으므로 0으로 설정
        'speedy': 0,  # 속도 데이터가 없으므로 0으로 설정
        'freeze_frame_360': freeze_frame,
        'success': 1 if row['outcome_name'] == 'success' else 0
    }
    return sample
if __name__=="__main__":

    final_df= pd.read_csv("total_data_with_state_label_mask.csv", index_col=0)
    row_data = final_df.iloc[0]
    print(row_data)
    sample = convert_row_to_sample(row_data)
    print(sample)
    print(len(sample['freeze_frame_360']))
    print(sample['freeze_frame_360'])
    tensor_converter = ToSoccerMapTensor()
    matrix, mask, target = tensor_converter(sample)

    print(matrix.shape) # (7, 68, 104)
    print(mask.shape) # (1, 68, 104)
    print(target)# tensor([1.]) or tensor([0.])
