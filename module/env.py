from utils import *
from rowtostate import *
class SoccerEnv(gym.Env):
    def __init__(self, dataset):
        super(SoccerEnv, self).__init__()
        self.dataset = dataset
        self.current_idx = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(80, 120, 11), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([80, 120])

    def reset(self):
        self.current_idx = 0
        state, _, _ = self.dataset[self.current_idx]
        return state

    def step(self):
        if self.current_idx >= len(self.dataset):
            return None, 0, True, {}
        state, action, reward = self.dataset[self.current_idx]

        self.current_idx += 1
        done = self.current_idx >= len(self.dataset)

        return state, reward, done, {}

    def get_action(self):
        if self.current_idx >= len(self.dataset):
            return None
        _, action, _ = self.dataset[self.current_idx]
        return action

    def row_to_state(self, row):
        # row로부터 상태를 생성하는 로직을 여기에 추가
        my_select = RowToState(row)
        (
            my_select.distance("ball").distance("goal").angle("goalpost", "cosine").angle("goalpost", "sine")
            .angle("goalpost", "angle").angle("ball", "cosine").angle("ball", "sine").angle("ball", "angle")
            .player_coor(row, role="actor").player_coor(row, role="teammate").player_coor(row, role="opponent")
        )
        stacked_features = my_select.stack_features()
        return stacked_features
